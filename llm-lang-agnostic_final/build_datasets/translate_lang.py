
import pandas as pd
import transformers
import sys
import os
from dataclasses import dataclass
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import argparse
# from llamawrapper import load_unemb_only, LlamaHelper





os.environ["HF_TOKEN"] = "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"



def format_hi_messages(en_word):
    examples_prompt = " English: tree - हिन्दी: पेड़, वृक्ष, वनस्पति, दरख़्त \n English: idea - हिन्दी: विचार, योजना, अवधारणा, कल्पना \n English: music - हिन्दी: संगीत, सुर, गाना \n English: flower - हिन्दी: फूल, पुष्प, कुसुम, उत्तमांश \n  English: country - हिन्दी: देश, गांव, देहात, भूभाग, वतन  \n "

    
    new_prompt = f"\n English: {en_word} - हिन्दी: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to hindi  dont generate anything else. Gernerate all the synonyms in Hindi but number of synonyms generated should not be more than 5. Separate the synonyms with a comma.Synonyms should always be single words.' +new_prompt},
    ]



def format_ta_messages(en_word):
    examples_prompt = " English: star - தமிழ்: நட்சத்திரம், விண்மீன், நக்ஷத்திரம், நாத்தாங்கி \n English: idea - தமிழ்: யோசனை, கருத்து, எண்ணம், திட்டம் \n English: music - தமிழ்: இசை, சங்கீதம் \n English: movie - தமிழ்: திரைப்படம், சலனப்படம், ஓடும் படம் \n  English: country - தமிழ்: நாடு, நிலப்பகுதி, கிராமம், நாட்டுப்புறம், தேசம் \n "

    
    new_prompt = f"\n English: {en_word} - தமிழ்: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Tamil dont generate anything else. Gernerate all the synonyms in Tamil but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words.' +new_prompt},
    ]

def format_te_messages(en_word):
    examples_prompt = " English: star - తెలుగు: నక్షత్రం, నక్షత్రము, ప్రముఖ నటుడు \n English: idea - తెలుగు: ఆలోచన, ఉద్దేశము, మనసుకి తట్టిన భావము \n English: music - తెలుగు: సంగీతం, సంగీతము, సంగీత రచనలు \n English: movie - తెలుగు: సినిమా, చలన చిత్రము, చలన చిత్ర ప్రదర్శన \n  English: country - తెలుగు: దేశం, నాడు \n "

    
    new_prompt = f"\n English: {en_word} - తెలుగు: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Telugu dont generate anything else. Gernerate all the synonyms in Telugu but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words.' +new_prompt},
    ]

def format_gu_messages(en_word):
    examples_prompt = " English: star - ગુજરાતી: તારો, તારક, તારાંકિત કરવું \n English: idea - ગુજરાતી: વિચાર, મનમાં ઊઠતું ખ્યાલ, ઇરાદો, આશય, હેતુ \n English: music - ગુજરાતી: સંગીત, ગાયન, સંગીતકલા \n English: movie - ગુજરાતી: ફિલ્મ, ચલચિત્ર \n  English: country - ગુજરાતી: દેશ, રાષ્ટ્ર, ભૂમિભાગ, વતન, રાષ્ટ્રભૂમિ \n "

    
    new_prompt = f"\n English: {en_word} - ગુજરાતી: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Gujarati dont generate anything else. Gernerate all the synonyms in Gujarati but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words.' +new_prompt},
    ]

def format_ml_messages(en_word):
    examples_prompt = " English: star - മലയാളം: നക്ഷത്രം, നക്ഷതം, താരം \n English: idea - മലയാളം: ആശയം, ആലോചന \n English: music - മലയാളം: സംഗീതം, പാട്ട്, ഗാനം \n English: movie - മലയാളം: സിനിമ, ചലച്ചിത്രം, പടം \n  English: country - മലയാളം: രാജ്യം, നാട്, രാജം, ദേശം \n "

    
    new_prompt = f"\n English: {en_word} - മലയാളം: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Malayalam dont generate anything else. Gernerate all the synonyms in Malayalam but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words.' +new_prompt},
    ]






def main(input_lang = 'en', target_lang = 'hi'):
    
    prefix = "./data/"
    # model_id = 'meta-llama/Llama-2-7b-hf'
    # model_id = "google/gemma-2-9b-it"
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    # model_id = "sarvamai/sarvam-2b-v0.5"
    df_en_fr = pd.read_csv(f'{prefix}{input_lang}/word_translation.csv').reindex()
    df1 = df_en_fr

    # if target_lang  in df1.columns:
    #     return
    # else:
    df1[target_lang] = ""


    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )


    for idx, row in df1.iterrows():
        if target_lang == 'hi':
            messages = format_hi_messages(row["word_original"])
        if target_lang == 'ml':
            messages = format_ml_messages(row["word_original"])
        if target_lang == 'gu':
            messages = format_gu_messages(row["word_original"])
        if target_lang == 'ta':
            messages = format_ta_messages(row["word_original"])
        if target_lang == 'te':
            messages = format_te_messages(row["word_original"])
        

        outputs = pipeline(
            messages,
            max_new_tokens=200,

        )
        print(outputs)
        outputs1 = outputs[0]["generated_text"][-1]['content']
        
    
        print(outputs1)
        # t1 = outputs1.split()[-1].strip()
        # print(t1)
        
        outputs2 = outputs1.strip()
        df1.at[idx, target_lang] = [word.strip() for word in outputs2.split(',') if word.strip()]


    # df1.to_csv(f'./data/{target_lang}/word_translation.csv', index=False)
    df1.to_csv(f'./data/{input_lang}/word_translation.csv', index=False)

    # if input_lang == "en":
    #     df1.to_csv(f'./data/{target_lang}/word_translation.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject parameters into a Python script.")
    parser.add_argument('--input_lang', type=str, required=True, help='Input lang')
    parser.add_argument('--target_lang', type=str, required=True, help='Target lang')

    args = parser.parse_args()
    main(args.input_lang, args.target_lang)







