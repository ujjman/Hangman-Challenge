# 🎉 Hangman ML Solver 🎉

Welcome to the **Hangman ML Solver** project! This repository showcases a machine learning-based approach to tackle the classic Hangman game. By combining two powerful algorithms—**CatBoost** and **XGBoost**—and layering heuristic strategies on top, this solver predicts which letters are most likely to complete a hidden word. 

---

## 1. Introduction 🚀

This project implements a **Hangman guess function** that:
1. Uses **CatBoost** and **XGBoost** models to predict letters.
2. Leverages a **Logistic Regression** meta-classifier to combine predictions from both models.
3. Adds **heuristic approaches** (prefix/suffix detection, substring analysis, and vowel-first strategy) to refine guesses.

Overall, these components help the solver achieve higher accuracy compared to a single-model approach. During experimentation, the success rate reached around **59.7%** with partial data usage, and it can be improved further by training on the **full dataset**.

---

## 2. Key Features ✨

1. **Multi-Label Classification**:  
   - Each model (CatBoost & XGBoost) is trained to predict the presence/absence of each letter (a–z) in the target word.

2. **Ensemble Stacking**:  
   - A **Logistic Regression** layer sits on top of the two base models to make final predictions (stacked ensemble).

3. **Heuristics**:
   - **Prefix/Suffix Detection**: Common prefixes (e.g., *anti-*, *inter-*) and suffixes (e.g., *-tion*, *-ment*) boost letter guesses if partial matches are found.
   - **Substring Analysis**: Looks at the local context around missing letters, using dictionary frequencies.
   - **Vowel-First Strategy**: Prioritizes guessing vowels early (since vowels are likely to appear in English words).

4. **Parallel Training & GPU Support**:
   - Large dataset management with **CatBoost** optionally running on **GPU** for faster training.
   - **XGBoost** trained in parallel, one classifier per letter.

5. **Balanced Accuracy**:
   - Evaluations are done via **balanced accuracy**, helping to handle skewed data for rare letters like *z* or *q*.

---

## 3. Dataset Generation 📚

### 3.1 Dictionary and Word Encoding
- **Original Dictionary**: ~250,000 English words.
- **Encoding**:
  - Each word is converted into:
    1. An **80-length integer array** encoding prefix and suffix positions.
    2. A **26-dimensional binary array** indicating which letters appear in the word.

### 3.2 Parallel Dataset Creation
- **Script**: `create_dataset.py`
- **Process**:
  - Reads `words_250000_train.txt`.
  - Generates multiple instances from each word by creating combinations of unique letters.
  - Uses **prefix-suffix mirroring** for the 80-length features.
  - Outputs a Parquet file `final_data.parquet` with the final feature-label pairs.

### 3.3 Key Improvement
- Ensures **every unique letter combination** is represented, increasing the diversity of partial-word states.
- Creates a **robust** training dataset that improves generalization in Hangman scenarios.

---

## 4. Base Models: CatBoost & XGBoost 🤖

### 4.1 CatBoost Model
- **Script**: `training_catboost.py`
- **Details**:
  1. **Multi-Label Setup**: 26 separate CatBoost classifiers (one per letter).
  2. **Training**: Uses a **train_test_split** (1% test) on the generated features from `final_data.parquet`.
  3. **Balanced Accuracy**: Measures performance for each letter.

### 4.2 XGBoost Model
- **Script**: `training_xgboost.py`
- **Details**:
  1. **Multi-Label Setup**: 26 separate XGBoost classifiers in parallel.
  2. **Checkpointing**: Training progress is saved iteratively to handle interruptions.
  3. **Evaluation**: Average balanced accuracy across all letters.

### 4.3 Modifications to Internet-Sourced Code
- **Data Sampling**: Enhanced technique to tackle class imbalance (rare letters).
- **Balanced Accuracy Metric**: More fair measure for skewed distributions.
- **GPU Support**: For CatBoost (`"task_type": "GPU"`) to speed up training.

---

## 5. Ensemble Approach 🔗

### 5.1 Ensemble Model Creation
- **Script**: `create_ensemble_model.py`
- **Process**:
  - Loads both **CatBoost** and **XGBoost** trained models.
  - Stacks their output probabilities into a **52-dimensional** (26 from CatBoost + 26 from XGBoost) feature array.

### 5.2 Meta-Classifier (Logistic Regression)
- **Strategy**:
  - For each letter, a **Logistic Regression** model learns to best combine the two model outputs.
  - Saves these **meta-classifiers** in `ensemble_model_xgb_catboost.pkl`.

### 5.3 Advantages
- **Complementary Strengths**: Leverages the strengths of CatBoost (categorical data handling) and XGBoost (gradient boosting efficiency).
- **Stacking Over Simple Averaging**: Enables a more **nuanced** combination for higher accuracy.

---

## 6. Heuristic Improvements 🧩

1. **Substring Analysis**:
   - `create_substrings` function checks local contexts (e.g., a window of 6 chars).
   - If a substring strongly correlates with a particular letter, it **boosts** that letter’s probability.

2. **Prefix & Suffix Detection**:
   - **Common Prefixes**: *anti-*, *inter-*, *tele-*, etc.
   - **Common Suffixes**: *-tion*, *-ment*, *-able*, etc.
   - If a high percentage of known prefix/suffix patterns match partially, missing letters from that pattern are guessed.

3. **Vowel-First Heuristic**:
   - Guesses vowels **early** (up to 2 for words ≤ 5 letters, up to 3 for words > 5 letters).

---


---

## 7. Results & Discussion 📈

### 7.1 Training Set Performance
- Balanced accuracy on frequent letters: **85–95%** range.
- Rare letters (e.g., *x*, *z*, *q*): slightly lower due to fewer training samples.

### 7.2 Testing on Disjoint Words
- Ensemble approach **generalizes** better than standalone models.
- Heuristics like prefix/suffix detection assist significantly with **longer words**.

### 7.3 Success Rate
- Achieved a **64%** success rate on ~500 tries with **partial training data**. 
- **Potential** for higher accuracy if trained on the **full `final_data.parquet`** dataset.

### 7.4 Future Improvements
1. **Full Dataset Training**: Avoid subsets for higher performance.
2. **Additional Heuristics**: Syllable-based segmentation, deeper context analysis.
3. **Advanced Model Architectures**: Possibly trying **transformer-based** embeddings for better letter predictions.

---

## 8. How to Run the Code ▶️

### 8.1 Prerequisites
- **Python 3.7+**  
- **Libraries** (install via pip):
  ```bash
  pip install numpy pandas catboost xgboost scikit-learn tqdm requests

- **GPU Drivers (Optional): If using CatBoost on a GPU** (task_type="GPU").
 
### 8.2 Execution Steps
- **Dataset Generation**

    ```bash
    python create_dataset.py
    ```
    Creates final_data.parquet.

- **Training CatBoost Model**

    ```bash
    python training_catboost.py
    ```
    Outputs multilabel_catboost_model.pkl.

- **Training XGBoost Model**

    ```bash
    python training_xgboost.py
    ```
    Outputs multilabel_xgb_model.pkl.

- **Creating Ensemble Model**

    ```bash
    python create_ensemble_model.py
    ```

    Outputs ensemble_model_xgb_catboost.pkl.

- **Running the Hangman Guess Function**

1. Open hangman_api_guess.ipynb in VSCode or Jupyter.
2. Run cells sequentially to interact with the Hangman API.