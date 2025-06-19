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


# target_lang = 'te'

os.environ["HF_TOKEN"] = "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"

prefix = "./data/langs/"


        

def main(input_lang = 'en', target_lang = 'hi'):

    

    # model_id = "meta-llama/Llama-3.3-70B-Instruct"
    df_en_fr = pd.read_csv(f'{prefix}{target_lang}/clean_cloze.csv').reindex()
    df = df_en_fr
    df1 = pd.read_csv(f'{prefix}{target_lang}/clean6.csv').reindex()


    


   

    for idx, row in df.iterrows():
        if row["word_original"] != df1.at[idx, "word_original"]:
            print('skipping...')
            continue
        df["word_translation"] = df1["word_translation"]

    
       


    df[["lang","word_original","word_translation","blank_prompt_translation_masked"]].to_csv(f'./data/langs/{target_lang}/clean_cloze.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject parameters into a Python script.")
    parser.add_argument('--input_lang', type=str, required=True, help='Input lang')
    parser.add_argument('--target_lang', type=str, required=True, help='Target lang')

    args = parser.parse_args()
    main(args.input_lang, args.target_lang)