import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from tqdm import tqdm  # Added import for tqdm

dataset = pd.read_parquet("final_data.parquet")

alpha = "abcdefghijklmnopqrstuvwxyz"

class MultiLabelCatBoostClassifier:
    def __init__(self, num_classes=26, catboost_params=None):
        # Initialize 26 CatBoost classifiers, one for each label
        self.classifiers = [CatBoostClassifier(iterations=1500, verbose=100) for _ in range(num_classes)]
    
    def fit(self, X, y):
        # X is the feature matrix, y is a binary matrix indicating label presence
        for i in tqdm(range(len(self.classifiers)), desc="Training classifiers"):  # Wrapped loop with tqdm
            # Train each classifier on the corresponding label
            # dd = y[y[alpha[i]] == 0].sample(len(y) - 2*len(y[y[alpha[i]] == 1])).index
            dd = []
            self.classifiers[i].fit(X.drop(dd), y[alpha[i]].drop(dd), verbose=100)
    
    def predict(self, X):
        # Predict probabilities for each label
        predictions = np.zeros((len(X), len(self.classifiers)))
        for i, clf in enumerate(self.classifiers):
            predictions[:, i] = clf.predict_proba(X)[:, 1]  # Probability of class '1'
        return predictions
    
    def save(self, filename):
        # Save the model to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    @classmethod
    def load(cls, filename):
        # Load the model from a pickle file
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model

model = MultiLabelCatBoostClassifier()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    dataset[[str(x) for x in range(80)]], 
    dataset[[x for x in "abcdefghijklmnopqrstuvwxyz"]], 
    test_size=0.01, 
    random_state=42
)

model.fit(X_train, y_train)

# Calculate accuracy between individual columns
def accuracy(y_pred, y_test):
    cols = y_test.columns
    x = []
    for i in range(len(cols)):
        print(cols[i], " : ", confusion_matrix(np.round(y_pred[:,i]), y_test[cols[i]]))
        print(cols[i], " : ", balanced_accuracy_score(np.round(y_pred[:,i]), y_test[cols[i]]))
        x.append(balanced_accuracy_score(np.round(y_pred[:,i]), y_test[cols[i]]))
    return np.mean(x)

accuracy(model.predict(X_test), y_test)

model.save('multilabel_catboost_model.pkl')

