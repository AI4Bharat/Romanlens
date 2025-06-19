
import subprocess

input_langs =["ml","ta","te","gu"]


for input_lang in input_langs:

    command = f" python build_datasets/translation_lang_translit_indic.py --input_lang {input_lang}"
    
                    
    # command = f" python build_datasets/translation_lang_translit.py --input_lang {input_lang}"
    subprocess.run(command, shell=True, check=True)
    
                    

print("All combinations processed!")