
import subprocess

input_langs =["hi","ml","ta","te","gu"]


for input_lang in input_langs:
    
                    
    command = f" python data_translation/translate_lang_translit.py --input_lang {input_lang}"
    subprocess.run(command, shell=True, check=True)
    
                    

print("All combinations processed!")