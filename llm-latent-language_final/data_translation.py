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
from llamawrapper import load_unemb_only, LlamaHelper
import seaborn as sns
from scipy.stats import bootstrap
from utils import plot_ci, plot_ci_plus_heatmap
from tqdm import tqdm

os.environ["HF_TOKEN"] = "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"

input_lang = 'fr'
prefix = "./data/langs/"
# model_id = 'meta-llama/Llama-2-7b-hf'
model_id = "google/gemma-2-9b-it"
df_en_fr = pd.read_csv(f'{prefix}{input_lang}/clean.csv').reindex()
df1 = df_en_fr

df1["lang"] = 'ml'
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
    examples_prompt = " English: mars - മലയാളം: ചൊവ്വ \n English: bottle - മലയാളം: കുപ്പി \n English: tongue - മലയാളം: നാവ് \n English: silver - മലയാളം: വെള്ളി \n "

    
    new_prompt = f"\n English: {en_word} - മലയാളം: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to malayalam dont generate anythiing else' +new_prompt},
    ]
        


for idx, row in df1.iterrows():
    messages = format1_messages(row["word_original"])

    outputs = pipeline(
        messages,
        max_new_tokens=64,

    )
    print(outputs)
    outputs1 = outputs[0]["generated_text"][-1]['content']
    
   
    print(outputs1)
    t1 = outputs1.split()[-1].strip()
    print(t1)
    
    df1.at[idx, "word_translation"] = t1.strip()
    


df1[["lang","word_original","word_translation"]].to_csv('./data/langs/ml/clean.csv', index=False)

