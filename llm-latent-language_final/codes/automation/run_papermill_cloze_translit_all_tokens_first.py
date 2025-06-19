import subprocess
# ["en","hi","ml","ta","te","gu"]
# ["ml_translit", "hi_translit", "ta_translit","te_translit","gu_translit","hi","ml","ta","te","gu"]
# List of input and target languages
# input_langs = [ "fr","de", "hi", "ml"]
target_langs = ["ml","ta","te","gu"]

# Loop through all combinations
# run from gu - gu_translit onwards
# "sarvamai/sarvam-2b-v0.5","google/gemma-2-9b-it""bigscience/bloom-7b1",

for target_lang in target_langs:
    
            for model in ["google/gemma-2-9b-it","google/gemma-2-9b","meta-llama/Llama-2-13b-chat-hf","meta-llama/Llama-2-7b-hf","mistralai/Mistral-7B-v0.1"]:
                
                model1 = ''
                if model == "google/gemma-2-9b-it":
                    model1 = 'google'
                elif model == "meta-llama/Llama-2-7b-hf":
                    model1 = 'llama_7b'
                var1 = ''
                
                print(f"Processing: Target - {target_lang}")
                command = f" papermill codes/cloze_translit_all_tokens_first.ipynb out.ipynb  -p target_lang {target_lang}   -p custom_model {model}"
                subprocess.run(command, shell=True, check=True)
                

print("All combinations processed!")