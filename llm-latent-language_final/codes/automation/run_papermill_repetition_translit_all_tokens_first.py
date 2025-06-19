

import subprocess
# ["en","hi","ml","ta","te","gu"]
# "fr","de", "hi", "ml"
# ["ml_translit", "hi_translit", "ta_translit","te_translit","gu_translit","hi","ml","ta","te","gu"]
# List of input and target languages
input_langs = ["hi","ml","ta","te","gu","ka","zh"]
target_langs =  ["hi","ml","ta","te","gu","ka","zh"]
shots = [6]
quant = ['true']
# Loop through all combinations
# run from gu - gu_translit onwards
# "sarvamai/sarvam-2b-v0.5","google/gemma-2-9b-it""bigscience/bloom-7b1","google/gemma-2-9b",,"google/gemma-2-9b","meta-llama/Llama-2-13b-chat-hf","meta-llama/Llama-2-7b-hf"
for model in ["mistralai/Mistral-7B-v0.1"]:
    for input_lang in input_langs:
        for target_lang in target_langs:
            
            for shot in shots:
                for X in quant:
                
                    if input_lang != target_lang  :
                        print(f"Skipping: Input - {input_lang}, Target - {target_lang} (same language)")
                        continue
                    model1 = ''
                    if model == "google/gemma-2-9b":
                        model1 = 'gemma-2-9b'
                    if model == "google/gemma-2-9b-it":
                        model1 = 'gemma-2-9b-it'
                    if model == "meta-llama/Llama-2-13b-chat-hf":
                        model1 = "Llama-2-13b-chat-hf"
                    elif model == "meta-llama/Llama-2-7b-hf":
                        model1 = 'llama_7b'
                    
                    var1 = ''
                    if X == 'true':
                        var1 = '8bit'
                    else:
                        var1 = '4bit'
                    print(f"Processing: Input - {input_lang}, Target - {target_lang}")
                    command = f" papermill codes/translation_translit_all_tokens_first.ipynb out.ipynb -p input_lang {input_lang} -p target_lang {target_lang} -p shots {shot} -p X {X} -p custom_model {model}"
                    subprocess.run(command, shell=True, check=True)
                    

print("All combinations processed!")