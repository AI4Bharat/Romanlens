import subprocess
# ["en","hi","ml","ta","te","gu"]
# ["ml_translit", "hi_translit", "ta_translit","te_translit","gu_translit","hi","ml","ta","te","gu"]


target_langs = ["ml","ta","te","gu"]



for target_lang in target_langs:
    
            for model in ["google/gemma-2-9b-it","google/gemma-2-9b","meta-llama/Llama-2-13b-chat-hf","meta-llama/Llama-2-7b-hf","mistralai/Mistral-7B-v0.1"]:
                
               
                
                print(f"Processing: Target - {target_lang}")
                command = f" papermill codes/cloze_translit_all_tokens_first.ipynb out.ipynb  -p target_lang {target_lang}   -p custom_model {model}"
                subprocess.run(command, shell=True, check=True)
                

print("All combinations processed!")