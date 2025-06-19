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
target_lang = 'ml'
input_lang = 'fr'
prefix = "./data/langs/"
# model_id = 'meta-llama/Llama-2-7b-hf'
# model_id = "google/gemma-2-9b-it"
model_id = "meta-llama/Llama-3.3-70B-Instruct"
df_en_fr = pd.read_csv(f'{prefix}{input_lang}/clean.csv').reindex()
df1 = df_en_fr[:10]

df1["lang"] = target_lang + '_translit'
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

def format1_messages(en_word):
    examples_prompt = " English: mars - Hindi: mangal grah \n English: women - Hindi: aurat \n English: woman - Hindi: mahila \n English: man - Hindi: aadmee \n "

    
    new_prompt = f"\n English: {en_word} - Hindi: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to hindi word in latin script dont generate anythiing else' +new_prompt},
    ]

def format_hi_messages(en_word):
    examples_prompt = " English: tree - Hindi: ped, vrksh, vanaspati, darakht \n English: idea - Hindi: vichaar, yojana, avadhaarana, kalpana \n English: music - Hindi: sangeet, sur, gaana \n English: flower - Hindi: phool, pushp, kusum \n  English: country - Hindi: desh, gaanv, dehaat, bhoobhaag, vatan \n "

    
    new_prompt = f"\n English: {en_word} - Hindi: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Hindi word in latin script dont generate anythiing else. Gernerate all the synonyms in Hindi in latin script but number of synonyms generated should not be more than 5. Separate the synonyms with a comma' +new_prompt},
    ]
        
def format_ta_messages(en_word):
    examples_prompt = " English: star - Tamil: natcattiram, vinmin, naksattiram, nattanki \n English: idea - Tamil: yocanai, karuttu, eṇṇam, tiṭṭam \n English: music - Tamil: icai, cankitam \n English: movie - Tamil: tiraippaṭam, calaṉappaṭam, otum patam \n  English: country - Tamil: natu, nilappakuti, kiramam, nattuppuram, tecam \n "

    
    new_prompt = f"\n English: {en_word} - Tamil: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Tamil word in latin script dont generate anythiing else. Gernerate all the synonyms in Tamil in latin script but number of synonyms generated should not be more than 5. Separate the synonyms with a comma' +new_prompt},
    ]

def format_te_messages(en_word):
    examples_prompt = " English: star - Telugu: naksatram, naksatramu, pramukha natudu \n English: idea - Telugu: alocana, uddesamu, manasuki tattina bhavamu \n English: music - Telugu: sangītam, sangitamu, sangita racanalu \n English: movie - Telugu: sinima, calana citramu, calana citra pradarsana \n  English: country - Telugu: desam, nadu \n "

    
    new_prompt = f"\n English: {en_word} - Telugu: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Telugu in latin script dont generate anythiing else. Gernerate all the synonyms in Telugu in latin script but number of synonyms generated should not be more than 5. Separate the synonyms with a comma' +new_prompt},
    ]

def format_gu_messages(en_word):
    examples_prompt = " English: star - Gujarati: taro, taraka, tarankita karavum \n English: idea - Gujarati: vicara, Manamam uṭhatum khyala, irado, asaya,hetu \n English: music - Gujarati: sangita, gayana, sangitakala \n English: movie - Gujarati: philma, calacitra, otum patam \n  English: country - Gujarati: desa, rastra, bhumibhaga, vatana, rastrabhumi \n "

    
    new_prompt = f"\n English: {en_word} - Gujarati: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Gujarati word in latin script dont generate anythiing else. Gernerate all the synonyms in Gujarati in latin script but number of synonyms generated should not be more than 5. Separate the synonyms with a comma' +new_prompt},
    ]

def format_ml_messages(en_word):
    examples_prompt = " English: star - Malayalam: nakshathram, nakshatham, thaaram \n English: idea - Malayalam: aashayam, aalochana \n English: music - Malayalam: sangeetham, paattu, ganam \n English: movie - Malayalam: sinima, chalachitham, padam \n  English: country - Malayalam: rajyam, naadu, rajam, desham \n "

    
    new_prompt = f"\n English: {en_word} - Malayalam: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Malayalam word in latin script dont generate anythiing else. Gernerate all the synonyms in Malayalam in latin script but number of synonyms generated should not be more than 5. Separate the synonyms with a comma' +new_prompt},
    ]

    

def main(input_lang = 'en', target_lang = 'hi'):
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
            max_new_tokens=150,

        )
        print(outputs)
        outputs1 = outputs[0]["generated_text"][-1]['content']
        
    
        print(outputs1)
        # t1 = outputs1.split()[-1].strip()
        # print(t1)
        
        df1.at[idx, "word_translation"] = outputs1.strip()
        


    df1[["lang","word_original","word_translation"]].to_csv(f'./data/langs/{target_lang}_translit/clean_test.csv', index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject parameters into a Python script.")
    parser.add_argument('--input_lang', type=str, required=True, help='Input lang')
    parser.add_argument('--target_lang', type=str, required=True, help='Target lang')

    args = parser.parse_args()
    main(args.input_lang, args.target_lang)
