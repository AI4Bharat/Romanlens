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

# Indic xlit
from ai4bharat.transliteration import XlitEngine

# intializing the indic-en multilingual model and dictionaries (if rerank option is True)
e = XlitEngine( beam_width=10, rescore=False, src_script_type = "indic")
e1 = XlitEngine("hi", beam_width=10, rescore=False, src_script_type = "en")

 

    


def main(input_lang = 'ml'):
    
    prefix = "./data/"
    # model_id = 'meta-llama/Llama-2-7b-hf'
    model_id = "google/gemma-2-9b-it"
    # model_id = "sarvamai/sarvam-2b-v0.5"
    df_en_fr = pd.read_csv(f'{prefix}en/word_translation.csv').reindex()
    df1 = df_en_fr

    target_lang = input_lang +'_devnagari'
    df1[target_lang] = ''

    # if target_lang  in df1.columns:
    #     return
    # else:
    #     df1[target_lang] = ""

    
    for idx, row in df1.iterrows():
        translit_list = []
        


        print (f'type(row[input_lang]: {type(row[input_lang])}')
        print(f'(row[input_lang]: {row[input_lang]}')
        print(f'row[input_lang].split(","): {row[input_lang].split(",")}')
        input_text = row[input_lang].strip('[]').replace("'", "")
        print(f'input_text: {input_text}')
        for word in input_text.split(","):
            word = word.strip()
            if word:
                romanized_word = e.translit_word(word, input_lang, topk=1)
                # transliterate Hindi word 
                translit_word = e1.translit_word( romanized_word[0], topk=1)

                print(f'translit word: {translit_word}')
                # translit word: {'hi': ['पुस्तकम']}
                translit_word1 = translit_word['hi'][0].replace(".", "")
                print(f'translit word_1: {translit_word1}')
                translit_list.append(translit_word1)
        

        # print(translit_list)    
        translit_string =  ','.join(translit_list)   
        print(f' translit_list: {translit_list}')
        print(f' translit_string: {translit_string}')

    
        
        
        # [word.strip() for word in translit_string.split(',') if word.strip()]
        df1.at[idx, target_lang] = translit_list


    # df1.to_csv(f'./data/{target_lang}/word_translation.csv', index=False)
    df1.to_csv(f'./data/en/word_translation.csv', index=False)

    # if input_lang == "en":
    #     df1.to_csv(f'./data/{target_lang}/word_translation.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject parameters into a Python script.")
    parser.add_argument('--input_lang', type=str, required=True, help='Input lang')
    # parser.add_argument('--target_lang', type=str, required=True, help='Target lang')

    args = parser.parse_args()
    main(args.input_lang)



