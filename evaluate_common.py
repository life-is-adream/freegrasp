import os

# Disable proxies and use HF mirror to prevent SSLEOFError
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-35fc45d726354c7a835f17e993809fe2")
# Force PyTorch SDPA to NOT use flash_attn (cu123 flash-attn is incompatible with CUDA 12.4)
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"

import pandas as pd

from calculate_SR import calculate_accuracy
from molmo_eval import batch_process_molmo
from reasoning_eval import process_dataset


PARQUET_FILES = [
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
]
NPZ_DIR = "data/npz_file"
MOLMO_OUTPUT_DIR = "data/output/molmo_output"

SUBSET_FILTERS = {
    "easy_no_ambi": ("Easy", False),
    "easy_yes_ambi": ("Easy", True),
    "medium_no_ambi": ("Medium", False),
    "medium_yes_ambi": ("Medium", True),
    "hard_no_ambi": ("Hard", False),
    "hard_yes_ambi": ("Hard", True),
}


def load_dataset() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(path) for path in PARQUET_FILES])


def prepare_subset(df: pd.DataFrame, subset_name: str, selected_split: str) -> pd.DataFrame:
    if subset_name not in SUBSET_FILTERS:
        raise ValueError(f"Unknown subset_name: {subset_name}")

    difficulty, ambiguous = SUBSET_FILTERS[subset_name]
    split_df = df[df["split"] == selected_split]
    return split_df[(split_df["difficulty"] == difficulty) & (split_df["ambiguious"] == ambiguous)]


def evaluate_subset(subset_name: str, selected_split: str = "0") -> float:
    os.makedirs(MOLMO_OUTPUT_DIR, exist_ok=True)

    df = load_dataset()
    batch_process_molmo(df, NPZ_DIR, MOLMO_OUTPUT_DIR)

    selected_subset = prepare_subset(df, subset_name, selected_split)
    print(
        f"Total number of examples in [SPLIT {selected_split}] "
        f"of [SET {subset_name}]: {len(selected_subset)}"
    )

    output_json_path = f"data/output/out_{subset_name}.json"
    process_dataset(selected_subset, NPZ_DIR, MOLMO_OUTPUT_DIR, output_json_path)
    return calculate_accuracy(output_json_path)
