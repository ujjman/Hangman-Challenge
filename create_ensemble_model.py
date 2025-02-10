
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


class MultiLabelCatBoostClassifier:
    """
    Trains 26 CatBoost binary classifiers (one per letter).

    Each classifier predicts if a letter is 'hidden' (i.e., in the word but not revealed in the partial encoding).

    """
    def __init__(self, num_classes=26, catboost_params=None):
        if catboost_params is None:
            catboost_params = {
                "iterations": 1500,
                "task_type": "GPU",
                "verbose": False
            }
        # Create 26 CatBoost classifiers
        self.classifiers = [
            CatBoostClassifier(**catboost_params) for _ in range(num_classes)
        ]
        self.num_classes = num_classes

    def fit(self, X, y):
        for i in range(self.num_classes):
            self.classifiers[i].fit(X, y[alpha[i]], verbose=100)

    def predict_proba(self, X):
        """

        Return predicted probability for each of the 26 letters.
        Shape: (n_samples, 26)

        """
        num_classes=26
        predictions = np.zeros((len(X), num_classes))
        for i, clf in enumerate(self.classifiers):
            # Probability that letter i is actually hidden in the word
            predictions[:, i] = clf.predict_proba(X)[:, 1]
        return predictions

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
        
        
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

"""
train_ensemble.py

Loads both the CatBoost and XGBoost multi-label models, generates probability
predictions, and trains a meta-classifier (e.g., Logistic Regression) as an
ensemble. Saves the final ensemble model to ensemble_model_xgb_catboost.pkl.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

alphabet = "abcdefghijklmnopqrstuvwxyz"

# --- 1. Define a class to encapsulate the Ensemble ---
class MultiLabelEnsemble:
    def __init__(self, catboost_model_path, xgboost_model_path):
        # Load the pre-trained CatBoost & XGBoost multi-label models
        self.catboost_model = self._load_model(catboost_model_path)
        self.xgboost_model = self._load_model(xgboost_model_path)
        # We will train 26 meta-classifiers (one for each letter)
        self.meta_classifiers = [LogisticRegression() for _ in range(26)]

    def _load_model(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def fit(self, X, y):
        # Check if the predictions are already saved
        catboost_pred_file = "catboost_pred_proba.npy"
        xgboost_pred_file = "xgboost_pred_proba.npy"
        
        # Load or compute CatBoost predictions
        if os.path.exists(catboost_pred_file):
            catboost_pred_proba = np.load(catboost_pred_file)
        else:
            catboost_pred_proba = self.catboost_model.predict_proba(X)
            np.save(catboost_pred_file, catboost_pred_proba)

        # Load or compute XGBoost predictions
        if os.path.exists(xgboost_pred_file):
            xgboost_pred_proba = np.load(xgboost_pred_file)
        else:
            xgboost_pred_proba = self.xgboost_model.predict_proba(X)
            np.save(xgboost_pred_file, xgboost_pred_proba)

        # Build stacked feature set => shape (n_samples, 52)
        stacked_features = np.concatenate([catboost_pred_proba, xgboost_pred_proba], axis=1)
        last_trained_index = -1  # Default, no classifiers trained

        # Train each meta-classifier with progress display
        meta_classifier_file = "meta_classifiers.pkl"
        if os.path.exists(meta_classifier_file):
            with open(meta_classifier_file, 'rb') as f:
                self.meta_classifiers = pickle.load(f)
                last_trained_index = 20

        for i in tqdm(range(last_trained_index + 1, 26), desc='Training Meta-classifiers'):
            letter = alphabet[i]
            y_i = y[letter].values
            self.meta_classifiers[i].fit(stacked_features, y_i)
            # Save the state of all trained classifiers
            with open(meta_classifier_file, 'wb') as f:
                pickle.dump(self.meta_classifiers, f)
            # Update the checkpoint index

    def predict_proba(self, X):
        catboost_pred_proba = self.catboost_model.predict_proba(X)
        xgboost_pred_proba = self.xgboost_model.predict_proba(X)

        stacked_features = np.concatenate([catboost_pred_proba, xgboost_pred_proba], axis=1)

        # Generate final probabilities for each letter
        final_preds = []
        for i in tqdm(range(26), desc='Predicting Probabilities'):
            probs_i = self.meta_classifiers[i].predict_proba(stacked_features)[:, 1]
            final_preds.append(probs_i)
        return np.array(final_preds).T

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

# --- 2. Train the ensemble model ---

def main():
    # Load the same dataset used for training.
    dataset = pd.read_parquet("final_data.parquet")
    catboost_model_path = "multilabel_catboost_model.pkl"
    xgboost_model_path = "multilabel_xgb_model.pkl"

    feature_cols = [str(x) for x in range(80)]
    label_cols = list(alphabet)

    X = dataset[feature_cols]
    Y = dataset[label_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.01, random_state=42
    )

    ensemble_model = MultiLabelEnsemble(
        catboost_model_path="multilabel_catboost_model.pkl",
        xgboost_model_path="multilabel_xgb_model.pkl"
    )

    print("Fitting the ensemble meta-classifiers...")
    ensemble_model.fit(X_train, y_train)
    print("Done fitting the ensemble.")

    y_pred_proba = ensemble_model.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    scores = []
    for i, letter in enumerate(alphabet):
        bac = balanced_accuracy_score(y_test[letter].values, y_pred[:, i])
        scores.append(bac)
        print(f"Letter '{letter}': Balanced Accuracy = {bac:.3f}")
    print(f"Ensemble Mean Balanced Accuracy = {np.mean(scores):.3f}")

    ensemble_model.save("ensemble_model_xgb_catboost.pkl")
    print("Saved ensemble model to ensemble_model_xgb_catboost.pkl")

if __name__ == "__main__":
    main()
