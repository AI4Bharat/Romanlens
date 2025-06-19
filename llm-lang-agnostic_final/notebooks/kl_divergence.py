import pandas as pd
import numpy as np
from scipy.stats import entropy
import os
import argparse

def kl_divergence(p, q):
    """
    Calculate KL divergence between two probability distributions
    Handles zero probabilities by adding small epsilon
    """
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    
    # Normalize to ensure they sum to 1
    p = p / p.sum()
    q = q / q.sum()
    
    return entropy(p, q)

def main(csv_file,model_name):
    # Calculate KL divergence for each pair
    df = pd.read_csv(csv_file)
    kl_d = {}

    for column1 in df.columns:
        for column2 in df.columns:
            if column1 == column2:
                continue
            if column1.replace("_translit","") == column2:
                kl_divergence_value = kl_divergence(df[column1], df[column2])
                kl_d[f'{column1}_rel_to_{column2}'] = kl_divergence_value

    # Convert dictionary to DataFrame with specific column names
    df_kl = pd.DataFrame({
        'romanized_rel_to_native': list(kl_d.keys()),
        'kl_divergence': list(kl_d.values())
    })



    # Define output file path
    output_file = 'kl_divergence_results.csv'

    csv_loc = 'kl_divergence'

    os.makedirs(f'{os.path.join(csv_loc, model_name)}', exist_ok=True)
    filepath = f'{os.path.join(csv_loc, model_name)}/source.csv'

        

    # Check if file exists and update/create accordingly
    if os.path.exists(filepath):
        # Read existing file
        df_existing = pd.read_csv(filepath)
        
        # Update existing entries and add new ones
        df_merged = pd.concat([df_existing, df_kl], ignore_index=True)
        
        # Remove duplicates, keeping the latest entry
        df_merged = df_merged.drop_duplicates(
            subset=['romanized_rel_to_native'], 
            keep='last'
        )
        
        # Save updated DataFrame
        df_merged.to_csv(filepath, index=False)
    else:
        # Create new file
        df_kl.to_csv(filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject parameters into a Python script.")
    parser.add_argument('--csv_file', type=str, required=True, help='csv file')
    parser.add_argument('--model_name', type=str, required=True, help='model name')

    args = parser.parse_args()
    main(args.csv_file, args.model_name)
