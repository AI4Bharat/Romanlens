

import subprocess
# ["en","hi","ml","ta","te","gu"]
# "fr","de", "hi", "ml"
# ["ml_translit", "hi_translit", "ta_translit","te_translit","gu_translit","hi","ml","ta","te","gu"]
# List of input and target languages
input_langs = ["en"]
target_langs =  ["hi","ml","ta","te","gu","ka","zh"]
shots = [6]


# "google/gemma-2-9b-it","google/gemma-2-9b",,"google/gemma-2-9b","meta-llama/Llama-2-13b-chat-hf","meta-llama/Llama-2-7b-hf"
for model in ["mistralai/Mistral-7B-v0.1"]:
    for input_lang in input_langs:
        for target_lang in target_langs:
            
            for shot in shots:
                
                
                if input_lang == target_lang  :
                    print(f"Skipping: Input - {input_lang}, Target - {target_lang} (same language)")
                    continue
                
                print(f"Processing: Input - {input_lang}, Target - {target_lang}")
                command = f" papermill codes/translation_translit_all_tokens_first.ipynb out.ipynb -p input_lang {input_lang} -p target_lang {target_lang} -p shots {shot} -p custom_model {model}"
                subprocess.run(command, shell=True, check=True)
                    

print("All combinations processed!")