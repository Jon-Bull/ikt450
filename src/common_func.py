# common_func.py
# This file could contain reusable functions that are commonly needed across notebooks and scripts

import os
import pandas as pd

def load_dataset(file_path, delimiter=","):
    """Loads a dataset from a given file path."""
    return pd.read_csv(file_path, delimiter=delimiter)

def save_dataframe(df, file_path):
    """Saves a DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

def ensure_dir_exists(directory):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)
