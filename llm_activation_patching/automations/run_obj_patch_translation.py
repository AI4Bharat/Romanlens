import subprocess
import json

# List of input and target languages

shots = [5]



translit_ins = [ "hi_translit","ml_translit", "ta_translit","te_translit","gu_translit"]

# translit_ins_1 = [ "ml_translit","hi_translit", "ta_translit","te_translit","gu_translit"]
indic_ins = ["hi", "ml", "ta", "te","gu"]
paper_args = [

    [[(l, "it") for l in translit_ins], "ml", "it"],
    [[(l, "it") for l in indic_ins], "ml", "it"],

]
# for pargs in paper_args:
#     object_patching_plot(*pargs, extra_langs=["en"])
# "google/gemma-2-9b-it","google/gemma-2-9b",
# ,"google/gemma-2-9b","mistralai/Mistral-7B-v0.1","meta-llama/Llama-2-7b-hf","meta-llama/Llama-2-13b-hf"

for model in [ "google/gemma-2-9b" ]:
    
    for pargs in paper_args:
        
        for sho in shots:
            
            
               
                model1 = ''
                if model == "google/gemma-2-9b-it":
                    model1 = 'gemma_2_9b_it'
                if model == "google/gemma-2-9b":
                    model1 = 'gemma_2_9b'

                if model == "mistralai/Mistral-7B-v0.1":
                    model1 = 'mistral_7b' 
                
                elif model == "meta-llama/Llama-2-7b-hf":
                    model1 = 'llama2_7b'
                s = ''
                for x in pargs:
                    if isinstance(x, list):
                        for y in x:
                            s+= f"{y[0]}_{y[1]}_"
                    else:
                        s+= f'{x}_'
                pargs_str = json.dumps(pargs)

              
                command = f" papermill notebooks/obj_patch_translation_1.ipynb out_dir/obj_patch_translation_1/{model1}_{s}_shots_{shots}.ipynb -p model {model} -p paper_args_str '{pargs_str}'"

                subprocess.run(command, shell=True, check=True)

    

                
                    

print("All combinations processed!")