import subprocess
import json
# ["en","hi","ml","ta","te","gu"]
# "fr","de", "hi", "ml"
# ["ml_translit", "hi_translit", "ta_translit","te_translit","gu_translit","hi","ml","ta","te","gu"]
# List of input and target languages
# input_langs = ["en" ,"it", "de", "zh", "fr","ml_translit", "hi_translit", "ta_translit","te_translit","gu_translit","ml", "ta","te","gu","hi"]
# target_langs = ["ml", "ta","te","gu","hi"]
shots = [5]
quant = ['true']
# Loop through all combinations
# run from gu - gu_translit onwards
# "sarvamai/sarvam-2b-v0.5","google/gemma-2-9b-it""bigscience/bloom-7b1",
# Open a file to log errors



translit_ins = [ "hi_translit","ml_translit", "ta_translit","te_translit","gu_translit"]

translit_ins_1 = [ "ml_translit","hi_translit", "ta_translit","te_translit","gu_translit"]
indic_ins = ["hi", "ml", "ta", "te","gu"]
paper_args = [

    [[(l, "it") for l in translit_ins], "ml", "it"],
    [[(l, "it") for l in indic_ins], "ml", "it"],

]
# for pargs in paper_args:
#     object_patching_plot(*pargs, extra_langs=["en"])
# "google/gemma-2-9b-it","google/gemma-2-9b","google/gemma-2-2b-it","meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B",
# ,"google/gemma-2-9b","meta-llama/Llama-3.1-8B-Instruct","mistralai/Mistral-7B-v0.1","meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-13b-hf"

for model in [ "google/gemma-2-9b" ]:
    i = 0
    for pargs in paper_args:
        i+=1
        for sho in shots:
            for X in quant:
            
               
                model1 = ''
                if model == "google/gemma-2-9b-it":
                    model1 = 'gemma_2_9b_it'
                if model == "google/gemma-2-9b":
                    model1 = 'gemma_2_9b'

                if model == "mistralai/Mistral-7B-v0.1":
                    model1 = 'mistral_7b'
                
                if model == "google/gemma-2-2b-it":
                    model1 = 'gemma_2_2b-it'
                
                if model == "google/gemma-2-2b":
                    model1 = 'gemma_2_2b'
                    
                if model == "meta-llama/Llama-3.1-8B":
                    model1 = 'Llama-3.1-8B'

                if model == "meta-llama/Llama-3.1-8B-Instruct":
                    model1 = 'Llama-3.1-8B-Instruct'

                if model == "meta-llama/Llama-3.2-3B":
                    model1 = 'Llama-3.2-3B'

                if model == "meta-llama/Llama-3.2-3B-Instruct":
                    model1 = 'Llama-3.2-3B-Instruct'
                
                
                
                elif model == "meta-llama/Llama-2-7b-hf":
                    model1 = 'llama_7b'
                s = ''
                for x in pargs:
                    if isinstance(x, list):
                        for y in x:
                            s+= f"{y[0]}_{y[1]}_"
                    else:
                        s+= f'{x}_'
                pargs_str = json.dumps(pargs)

                remove_model = 'no'

                if i ==4:
                    remove_model == 'yes'

              
                command = f" papermill notebooks/obj_patch_translation_1.ipynb out_dir/obj_patch_translation_1/{model1}_{s}_shots_{shots}.ipynb -p model {model} -p paper_args_str '{pargs_str}' -p del_model {remove_model} "
                # command = f" papermill notebooks/obj_patch_translation_1.ipynb out.ipynb -p model {model} -p paper_args_str '{pargs_str}' -p del_model {remove_model} "
                subprocess.run(command, shell=True, check=True)

    

                
                    

print("All combinations processed!")