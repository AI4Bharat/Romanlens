import subprocess
# ["en","hi","ml","ta","te","gu"]
# ["ml_translit", "hi_translit", "ta_translit","te_translit","gu_translit","hi","ml","ta","te","gu"]
# List of input and target languages
input_langs = ["en", "fr","de" ] 
target_langs = ["hi","hi_translit","ml","ml_translit","ta","ta_translit","te","te_translit","gu","gu_translit"]
shots = [6]
# six shots would mean 5 examples and translate the sixth word


for model in ["google/gemma-2-9b-it","google/gemma-2-9b","meta-llama/Llama-2-13b-chat-hf","meta-llama/Llama-2-7b-hf"]:
    for input_lang in input_langs:
        for target_lang in target_langs:
            for shot in shots:
                
                
                if input_lang == target_lang :
                    print(f"Skipping: Input - {input_lang}, Target - {target_lang} (same language)")
                    continue
                
                print(f"Processing: Input - {input_lang}, Target - {target_lang}")
                # out/out_{model1}_{input_lang}_{target_lang}_{shot}_{var1}_dummy.ipynb
                command = f"papermill translation2.ipynb out.ipynb -p input_lang {input_lang} -p target_lang {target_lang} -p shots {shot} -p custom_model {model}"
                subprocess.run(command, shell=True, check=True)
                    

print("All combinations processed!")