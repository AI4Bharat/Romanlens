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

input_lang = 'ml'
prefix = "./data/langs/"
# model_id = 'meta-llama/Llama-2-7b-hf'
# model_id = "google/gemma-2-9b-it"



# Indic xlit
from ai4bharat.transliteration import XlitEngine

# intializing the indic-en multilingual model and dictionaries (if rerank option is True)
e = XlitEngine( beam_width=10, rescore=False, src_script_type = "indic")

    

def main(input_lang = 'ml'):

    df_en_fr = pd.read_csv(f'{prefix}{input_lang}/clean6.csv').reindex()

    df1 = df_en_fr.copy()

    df1["lang"] = input_lang + '_translit'
    df1["word_translation"] = ""


    for idx, row in df_en_fr.iterrows():
        translit_list = []
        print(f'row["word_translation"].split(","): {row["word_translation"].split(",")}')
        for word in row["word_translation"].split(","):
            word = word.strip()
            if word:
                translit_word = e.translit_word(word, input_lang, topk=1)
                print(f'translit word: {translit_word}')
            
                translit_word1 = translit_word[0].replace(".", "")
                print(f'translit word_1: {translit_word1}')
                translit_list += translit_word

            
        translit_string =  ','.join(translit_list)   
        print(f' translit_string: {translit_string}')
        df1.at[idx, "word_translation"] = translit_string
            


    df1[["lang","word_original","word_translation"]].to_csv(f'./data/langs/{input_lang}_translit/clean6.csv', index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject parameters into a Python script.")
    parser.add_argument('--input_lang', type=str, required=True, help='Input lang')
   

    args = parser.parse_args()
    main(args.input_lang)
