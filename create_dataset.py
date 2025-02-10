"""

Generates the Hangman dataset and saves it as a Parquet file (final_data.parquet).
We will use a similar approach as the CatBoost dataset creation,
but the final dataset can be used by any model (XGBoost, CatBoost, etc.).
"""

import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
import multiprocessing as mp

# Define letter-value mapping
value_map = {
    "a": 1,  "b": 2,  "c": 3,  "d": 4,  "e": 5,  "f": 6,
    "g": 7,  "h": 8,  "i": 9,  "j": 10, "k": 11, "l": 12,
    "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18,
    "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24,
    "y": 25, "z": 26
}
alphabet = list(value_map.keys())

def build_dictionary(dictionary_file_location):
    """Read dictionary words from a text file."""
    with open(dictionary_file_location, "r") as f:
        full_dictionary = f.read().splitlines()
    return full_dictionary

def get_combinations(letters):
    """Generate all non-empty combinations of letters from a set."""
    combinations = []
    for r in range(1, len(letters) + 1):
        combinations += list(itertools.combinations(letters, r))
    return combinations

def dataframemaker(dicc):
    """
    For each word in dicc:
      1. Get all combinations of unique letters in that word.
      2. Create an 80-length feature encoding (prefix + suffix).
      3. Create 26 binary labels (does word contain letter?).
      4. Return rows: [80 encoded positions] + [26 binary labels].
    """
    rows = []
    for word in dicc:
        unique_letters = "".join(set(word))  # set to remove duplicates
        combs = get_combinations(unique_letters)
        k = len(word)

        # For each combination of letters in 'word'
        for comb in combs:
            comb_set = set(comb)

            # Create an array of length 80, default -1
            feat_arr = np.full(80, -1, dtype=np.int8)

            # Create the 26-label array (which letters to guess = 1 or 0)
            label_arr = np.zeros(26, dtype=np.int8)

            # Fill feature array: for positions that are in the comb, store letter value; 
            # else store 0. Mirror these from the front and back to encode prefix/suffix.
            for i, ch in enumerate(word):
                if ch in comb_set:
                    feat_arr[i] = value_map[ch]
                    feat_arr[80 - k + i] = value_map[ch]
                else:
                    feat_arr[i] = 0
                    feat_arr[80 - k + i] = 0
                    label_arr[value_map[ch] - 1] = 1  # Means we "need" to guess this letter

            # rows = 80 features + 26 label columns
            rows.append(list(feat_arr) + list(label_arr))

    df = pd.DataFrame(rows, dtype=np.int8)
    return df

def main():
    # 1. Loading the training dictionary
    train_dict = build_dictionary("words_250000_train.txt")

    # 2. Split into chunks and process in parallel to build the dataset.
    num_splits = 100
    splitted_dicts = np.array_split(train_dict, num_splits)

    df_list = []
    for i, dic_chunk in enumerate(splitted_dicts, start=1):
        print(f"Processing chunk {i}/{num_splits} ...")
        manager = mp.Manager()
        pool = mp.Pool()

        sub_splits = np.array_split(dic_chunk, 128)
        partial_dfs = pool.map(dataframemaker, sub_splits)

        pool.close()
        pool.join()

        chunk_df = pd.concat(partial_dfs, ignore_index=True)
        df_list.append(chunk_df)

    final_df = pd.concat(df_list, ignore_index=True)

    # 3. Assign column names: 0..79 for features + a..z for labels
    feature_cols = [str(x) for x in range(80)]
    label_cols = alphabet
    final_df.columns = feature_cols + label_cols

    # 4. Save to Parquet for efficient loading
    final_df.to_parquet("final_data.parquet")
    print("Saved dataset to final_data.parquet")

if __name__ == "__main__":
    main()
