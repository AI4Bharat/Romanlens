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

# Make sure bitsandbytes is available for 4-bit quantization
# from bitsandbytes import quantize

# HF_HOME=/data/huggingface_cache
target_lang = 'ml'
os.environ["HF_TOKEN"] = "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"

# os.environ['HF_HOME'] = 'home/alan/huggingface_cache'

# # Verify that the environment variable has been set
# print(os.environ['HF_HOME'])


input_lang = 'en'
prefix = "./data/langs/"
# model_id = 'meta-llama/Llama-2-7b-hf','google/gemma-2-9b-it'
model_id = "meta-llama/Llama-3.3-70B-Instruct"
df_en_fr = pd.read_csv(f'{prefix}{input_lang}/clean.csv').reindex()
df1 = df_en_fr

df1["lang"] = target_lang
df1["word_translation"] = ""


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    
)

# def format_messages(en_word):
#     examples_prompt = " English: mars - മലയാളം: ചൊവ്വ \n English: bottle - മലയാളം: കുപ്പി \n English: tongue - മലയാളം: നാവ് \n English: silver - മലയാളം: വെള്ളി \n "

    
#     new_prompt = f"\n English: {en_word} - മലയാളം: "
    
#     return  examples_prompt + new_prompt

# def format1_messages(en_word):
#     examples_prompt = " English: mars - हिन्दी: मंगल ग्रह \n English: bottle - हिन्दी: बोतल \n English: tongue - हिन्दी: जीभ \n English: silver - हिन्दी: चाँदी \n "

    
#     new_prompt = f"\n English: {en_word} - हिन्दी: "
    
#     return [
#     {"role": "user", "content":  examples_prompt +'Just translate the english word to hindi dont generate anythiing else' +new_prompt},
#     ]

def format_ka_messages(en_word):
    examples_prompt = """ English: tree - ქართული: ხე, მერქანი, მცენარე, ტყე 
 English: idea - ქართული: იდეა, მოსაზრება, შეხედულება, გეგმა 
 English: music - ქართული: მუსიკა, მელოდია, სიმღერა 
 English: flower - ქართული: ყვავილი, მცენარე, ფურცელი, სურნელოვანი 
 English: country - ქართული: ქვეყანა, სოფელი, დაბა, ტერიტორია, ვატანი 
 """
    
    new_prompt = f"\n English: {en_word} - ქართული: "

    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Georgian dont generate anything else. Gernerate all the synonyms in Georgian but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]

def format_am_messages(en_word):
    examples_prompt = """ English: tree - አማርኛ: ዛፍ, ተክል, እንጨት, ደን 
 English: idea - አማርኛ: ሃሳብ, ዕቅፍ, ሐሳብ, አስተሳሰብ 
 English: music - አማርኛ: ሙዚቃ, ዜማ, መዝሙር 
 English: flower - አማርኛ: አበባ, አበባዎች, ፍጠራ, አበባናምርናት 
 English: country - አማርኛ: አገር, ገጠር, ክልል, የአገርነት, ዋታንንት 
 """
    
    new_prompt = f"\n English: {en_word} - አማርኛ: "

    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Amharic dont generate anything else. Gernerate all the synonyms in Amharic but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]

    
def format_hi_messages(en_word):
    examples_prompt = " English: tree - हिन्दी: पेड़, वृक्ष, वनस्पति, दरख़्त \n English: idea - हिन्दी: विचार, योजना, अवधारणा, कल्पना \n English: music - हिन्दी: संगीत, सुर, गाना \n English: flower - हिन्दी: फूल, पुष्प, कुसुम, उत्तमांश \n  English: country - हिन्दी: देश, गांव, देहात, भूभाग, वतन  \n "

    
    new_prompt = f"\n English: {en_word} - हिन्दी: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to hindi  dont generate anything else. Gernerate all the synonyms in Hindi but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]

def format_he_messages(en_word):
    examples_prompt = " English: tree - עברית: עץ, אילן, צמח, יער \n English: idea - עברית: רעיון, מחשבה, מושג, תכנית \n English: music - עברית: מוזיקה, שירה, מנגינה, צליל \n English: flower - עברית: פרח, בלום, סגלגל, כותרת \n  English: country - עברית: מדינה, ארץ, מולדת, מחוז, אומה \n "
    
    new_prompt = f"\n English: {en_word} - עברית: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Hebrew  dont generate anything else. Gernerate all the synonyms in Hebrew but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]

def format_th_messages(en_word):
    examples_prompt = " English: tree - ภาษาไทย: ต้นไม้, ไม้, พืช, ป่า \n English: idea - ภาษาไทย: ความคิด, ไอเดีย, แนวคิด, ทฤษฎี \n English: music - ภาษาไทย: ดนตรี, เพลง, เสียง, ทำนอง \n English: flower - ภาษาไทย: ดอกไม้, มาลัย, ดอก \n  English: country - ภาษาไทย: ประเทศ, ชาติ, ดินแดน, เมือง, อาณาจักร \n "
    
    new_prompt = f"\n English: {en_word} - ภาษาไทย: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Thai dont generate anything else. Gernerate all the synonyms in Thai but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]




def format_ta_messages(en_word):
    examples_prompt = " English: star - தமிழ்: நட்சத்திரம், விண்மீன், நக்ஷத்திரம், நாத்தாங்கி \n English: idea - தமிழ்: யோசனை, கருத்து, எண்ணம், திட்டம் \n English: music - தமிழ்: இசை, சங்கீதம் \n English: movie - தமிழ்: திரைப்படம், சலனப்படம், ஓடும் படம் \n  English: country - தமிழ்: நாடு, நிலப்பகுதி, கிராமம், நாட்டுப்புறம், தேசம் \n "

    
    new_prompt = f"\n English: {en_word} - தமிழ்: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Tamil dont generate anythiing else. Gernerate all the synonyms in Tamil but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]

def format_te_messages(en_word):
    examples_prompt = " English: star - తెలుగు: నక్షత్రం, నక్షత్రము, ప్రముఖ నటుడు \n English: idea - తెలుగు: ఆలోచన, ఉద్దేశము, మనసుకి తట్టిన భావము \n English: music - తెలుగు: సంగీతం, సంగీతము, సంగీత రచనలు \n English: movie - తెలుగు: సినిమా, చలన చిత్రము, చలన చిత్ర ప్రదర్శన \n  English: country - తెలుగు: దేశం, నాడు \n "

    
    new_prompt = f"\n English: {en_word} - తెలుగు: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Telugu dont generate anythiing else. Gernerate all the synonyms in Telugu but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]

def format_gu_messages(en_word):
    examples_prompt = " English: star - ગુજરાતી: તારો, તારક, તારાંકિત કરવું \n English: idea - ગુજરાતી: વિચાર, મનમાં ઊઠતું ખ્યાલ, ઇરાદો, આશય, હેતુ \n English: music - ગુજરાતી: સંગીત, ગાયન, સંગીતકલા \n English: movie - ગુજરાતી: ફિલ્મ, ચલચિત્ર \n  English: country - ગુજરાતી: દેશ, રાષ્ટ્ર, ભૂમિભાગ, વતન, રાષ્ટ્રભૂમિ \n "

    
    new_prompt = f"\n English: {en_word} - ગુજરાતી: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Gujarati dont generate anythiing else. Gernerate all the synonyms in Gujarati but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]




def format_ml_messages(en_word):
    examples_prompt = " English: star - മലയാളം: നക്ഷത്രം, നക്ഷതം, താരം \n English: idea - മലയാളം: ആശയം, ആലോചന \n English: music - മലയാളം: സംഗീതം, പാട്ട്, ഗാനം \n English: movie - മലയാളം: സിനിമ, ചലച്ചിത്രം, പടം \n  English: country - മലയാളം: രാജ്യം, നാട്, രാജം, ദേശം \n "

    
    new_prompt = f"\n English: {en_word} - മലയാളം: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Malayalam dont generate anythiing else. Gernerate all the synonyms in Malayalam but number of synonyms generated should not be more than 5. Separate the synonyms with a comma. Synonyms should always be single words' +new_prompt},
    ]

def format_es_messages(en_word):
    examples_prompt = " English: star - español: estrella \n English: crime - español: delito \n English: music - español: música \n English: movie - español: película \n  English: country - español: país \n "

    
    new_prompt = f"\n English: {en_word} - español: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Spanish dont generate anythiing else.' +new_prompt},
    ]
def format_it_messages(en_word):
    examples_prompt = " English: star - Italiano: stella \n English: crime - Italiano: crimine \n English: music - Italiano: musica \n English: movie - Italiano: film \n  English: country - Italiano: paese \n "

    
    new_prompt = f"\n English: {en_word} - Italiano: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Italian dont generate anythiing else.' +new_prompt},
    ]


def format_da_messages(en_word):
    examples_prompt = " English: star - dansk: stjerne \n English: crime - dansk: forbrydelse \n English: music - dansk: musik \n English: sea - dansk: hav \n  English: beautiful - dansk: smuk \n "

    
    new_prompt = f"\n English: {en_word} - dansk: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Danish dont generate anythiing else.' +new_prompt},
    ]









        
def main(input_lang = 'en', target_lang = 'hi'):

    for idx, row in df1.iterrows():
        # 
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
        if target_lang == 'ka':
            messages = format_ka_messages(row["word_original"])
        if target_lang == 'am':
            messages = format_am_messages(row["word_original"])
        if target_lang == 'he':
            messages = format_he_messages(row["word_original"])
        if target_lang == 'th':
            messages = format_th_messages(row["word_original"])

        outputs = pipeline(
            messages,
            max_new_tokens = 150,

        )
        print(outputs)
        outputs1 = outputs[0]["generated_text"][-1]['content']
        
    
        print(outputs1)
        # t1 = outputs1.split()[-1].strip()
        # print(t1)
        
        df1.at[idx, "word_translation"] = outputs1.strip()
        


    df1[["lang","word_original","word_translation"]].to_csv(f'./data/langs/{target_lang}/clean6.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject parameters into a Python script.")
    parser.add_argument('--input_lang', type=str, required=True, help='Input lang')
    parser.add_argument('--target_lang', type=str, required=True, help='Target lang')

    args = parser.parse_args()
    main(args.input_lang, args.target_lang)

