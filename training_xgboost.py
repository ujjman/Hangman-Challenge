"""
Loads the final_data.parquet dataset, trains 26 parallel XGBoost classifiers
(one for each letter), and saves the model to multilabel_xgb_model.pkl.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pickle
from tqdm import tqdm
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pickle
from tqdm import tqdm
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pickle
from tqdm import tqdm
from typing import Optional, Dict, List
import gc  # Garbage collection


import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pickle
from tqdm import tqdm
from typing import Optional, Dict, List
import gc  # Garbage collection

# Define the alphabet globally
alphabet = "abcdefghijklmnopqrstuvwxyz"

class MultiLabelXGBoostClassifier:
    """
    Trains 26 separate XGBoost binary classifiers, each predicting whether
    a letter is needed in the guess set.
    Includes checkpointing to save progress after each classifier is trained.
    """
    def __init__(self, num_classes: int = 26, xgb_params: Optional[Dict] = None):
        self.num_classes = num_classes
        # Default XGBoost params can be overridden
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0  # Suppress XGBoost's own logs
        }
        if xgb_params is not None:
            default_params.update(xgb_params)
        self.xgb_params = default_params

        # Create a list of XGBClassifier models
        self.models: List[xgb.XGBClassifier] = [
            xgb.XGBClassifier(**self.xgb_params) for _ in range(num_classes)
        ]

        # Keep track of trained classifier indices
        self.trained_indices = set()

    def train_single_classifier(self, i: int, X: pd.DataFrame, y_i: np.ndarray):
        """
        Trains a single classifier for the given index.
        """
        letter = alphabet[i]
        print(f"Training classifier for letter '{letter}'...")
        self.models[i].fit(X, y_i)
        self.trained_indices.add(i)
        print(f"Classifier for letter '{letter}' trained.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns an array of shape (n_samples, 26) with the predicted probability
        for each letter being '1' (i.e., needed).
        """
        all_preds = []
        print("Generating probability predictions...")
        # Initialize tqdm progress bar
        progress_bar = tqdm(range(self.num_classes), desc="Predicting Probabilities", unit="classifier", dynamic_ncols=True)
        for i in progress_bar:
            letter = alphabet[i]
            progress_bar.set_description(f"Predicting '{letter}' Probability")
            preds_i = self.models[i].predict_proba(X)[:, 1]
            all_preds.append(preds_i)
        progress_bar.close()
        return np.array(all_preds).T  # shape => (n_samples, 26)

    def save(self, filename: str):
        """
        Saves the trained model to a file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename: str):
        """
        Loads a trained model from a file.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model

def main():
    import warnings
    warnings.filterwarnings("ignore")  # Suppress warnings for clean output

    # Define file paths
    dataset_path = "final_data.parquet"
    model_path = "multilabel_xgb_model.pkl"  # Change as needed

    # 1. Load dataset
    print("Loading dataset...")
    try:
        dataset = pd.read_parquet(dataset_path)
    except FileNotFoundError:
        print("Dataset not found at the specified path. Please check the path and try again.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # Define feature and label columns
    feature_cols = [str(x) for x in range(80)]
    label_cols = list(alphabet)

    # Ensure that all required columns are present
    missing_features = set(feature_cols) - set(dataset.columns)
    missing_labels = set(label_cols) - set(dataset.columns)
    if missing_features:
        print(f"Missing feature columns: {missing_features}")
        return
    if missing_labels:
        print(f"Missing label columns: {missing_labels}")
        return

    # Select features and labels
    X = dataset[feature_cols]
    Y = dataset[label_cols]


    # 2. Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.01, random_state=42
    )


    # 3. Initialize the multi-label XGBoost classifier
    
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "eval_metric": "logloss",
        "verbosity": 0  # Suppress XGBoost's own logs
    }
    model = MultiLabelXGBoostClassifier(xgb_params=xgb_params)

    # 4. Train classifiers incrementally with checkpointing
    print("Training XGBoost models")
    # Iterate through all classifiers
    for i in tqdm(range(model.num_classes), desc="Training Classifiers", unit="classifier", dynamic_ncols=True):

        letter = alphabet[i]
        y_i = y_train[letter].values

        # Update the tqdm description
        tqdm.write(f"Training classifier for letter '{letter}'...")
        # Train the single classifier
        model.train_single_classifier(i, X_train, y_i)


    print("All classifiers have been trained.")


    # 5. Evaluate using balanced_accuracy_score
    print("Evaluating model performance...")
    y_pred_proba = model.predict_proba(X_test)
    # Round probabilities to get 0/1 predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)

    scores = []
    # Initialize tqdm progress bar for evaluation
    progress_bar = tqdm(range(26), desc="Evaluating Classifiers", unit="classifier", dynamic_ncols=True)
    for i in progress_bar:
        letter = alphabet[i]
        progress_bar.set_description(f"Evaluating '{letter}' Classifier")
        bac = balanced_accuracy_score(y_test[letter].values, y_pred[:, i])
        scores.append(bac)
        tqdm.write(f"Letter '{letter}': Balanced Accuracy = {bac:.3f}")
    progress_bar.close()

    mean_bac = np.mean(scores)
    print(f"Mean Balanced Accuracy = {mean_bac:.3f}")

    # 6. Save final model
    model.save(model_path)
    print("Final model saved.")


if __name__ == "__main__":
    main()
