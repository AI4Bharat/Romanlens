import subprocess
import json
# "google/gemma-2-9b-it","meta-llama/Llama-3.1-8B-Instruct"
for model in [ "google/gemma-2-9b" ]:

    model_name = model.split("/")[-1]
    csv_file = f'csv_file/{model_name}/source_prob.csv'



    command = f" python notebooks/kl_divergence.py --csv_file {csv_file} --model_name {model_name}"                
    subprocess.run(command, shell=True, check=True)
                    
    
    


print("All combinations processed!")