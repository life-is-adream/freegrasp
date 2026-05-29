import os

# Disable proxies and use HF mirror to prevent SSLEOFError
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-35fc45d726354c7a835f17e993809fe2")
# Force PyTorch SDPA to NOT use flash_attn (cu123 flash-attn is incompatible with CUDA 12.4)
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"

import pandas as pd

from molmo_eval import batch_process_molmo 
from reasoning_eval import process_dataset
from calculate_SR import calculate_accuracy

def evaluate_pipeline():
    # Path of dataset
    PARQUET_FILES = [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet"
    ]
    NPZ_DIR = "data/npz_file"

    MOLMO_OUTPUT_DIR = "data/output/molmo_output"
    os.makedirs(MOLMO_OUTPUT_DIR, exist_ok=True)

    # Load the dataset
    df = pd.concat([pd.read_parquet(p) for p in PARQUET_FILES])
    batch_process_molmo(df, NPZ_DIR, MOLMO_OUTPUT_DIR)

    # Select split to run
    selected_split = '0'  # Change this value to 0, 1, or 2

    df = df[df['split'] == selected_split]  # Filter dataset based on selected split

    # Select subsets for testing
    easy_no_ambi = df[(df['difficulty'] == 'Easy') & (df['ambiguious'] == False)]
    easy_yes_ambi = df[(df['difficulty'] == 'Easy') & (df['ambiguious'] == True)]
    medium_no_ambi = df[(df['difficulty'] == 'Medium') & (df['ambiguious'] == False)]
    medium_yes_ambi = df[(df['difficulty'] == 'Medium') & (df['ambiguious'] == True)]
    hard_no_ambi = df[(df['difficulty'] == 'Hard') & (df['ambiguious'] == False)]
    hard_yes_ambi = df[(df['difficulty'] == 'Hard') & (df['ambiguious'] == True)]

    # Choose a subset for testing (e.g., easy_no_ambi)
    subset_name = "easy_yes_ambi"  
    selected_subset = eval(subset_name)  
    print(f"Total number of examples in [SPLIT {selected_split}] of [SET {subset_name}]: {len(selected_subset)}")

    # Run test
    dataset_root = NPZ_DIR
    output_json_path = f"data/output/out_{subset_name}.json"
    process_dataset(selected_subset, dataset_root, MOLMO_OUTPUT_DIR, output_json_path)

    sr = calculate_accuracy(output_json_path)


if __name__ == "__main__":
    evaluate_pipeline()