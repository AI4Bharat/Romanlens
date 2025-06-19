
import subprocess
# ["en","hi","ml","ta","te","gu"]
# "fr","de", "hi", "ml"
# ["ml_translit", "hi_translit", "ta_translit","te_translit","gu_translit","hi","ml","ta","te","gu"]
# List of input and target languages
input_langs = ["en"]
target_langs = ["th"]
shots = [6]
quant = ['true']
# Loop through all combinations

# Open a file to log errors

for input_lang in input_langs:
    for target_lang in target_langs:
        
        for shot in shots:
            for X in quant:
                for model in ["meta-llama/Llama-3.3-70B-Instruct"]:
                    if input_lang == target_lang :
                        print(f"Skipping: Input - {input_lang}, Target - {target_lang} (same language)")
                        continue
                   
                    
                    
                    print(f"Processing: Input - {input_lang}, Target - {target_lang}")
                    # command = f" python data_translation/cloze_translate.py --input_lang {input_lang} --target_lang {target_lang}"
                    command = f" python data_translation/cloze_translate.py --input_lang {input_lang} --target_lang {target_lang}"
                    subprocess.run(command, shell=True, check=True)
                    
                    

print("All combinations processed!")