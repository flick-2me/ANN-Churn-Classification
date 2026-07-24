# Customer Churn Prediction using Artificial Neural Networks (ANN)

This project builds an end-to-end Machine Learning and Deep Learning pipeline to predict customer churn for a banking institution using TensorFlow/Keras and Streamlit.

---

## 📌 Project Overview

Customer retention is critical for banking operations. This repository contains the complete workflow for predicting whether a bank customer will exit (`Exited = 1`) or stay (`Exited = 0`) based on their demographic and financial attributes.

---

## 🛠️ Operations Performed in the Notebook (`experiments.ipynb`)

The notebook documents the experimental workflow, from raw data preprocessing to deep learning model execution:

### 1. Data Cleaning & Feature Engineering
* **Data Ingestion:** Loaded customer records from `Churn_Modelling.csv` using `pandas`[cite: 1, 2].
* **Identifier Removal:** Dropped non-predictive columns (`RowNumber`, `CustomerId`, `Surname`)[cite: 1, 2].
* **Categorical Encoding:**
  * **`Gender` Column:** Encoded using `sklearn.preprocessing.LabelEncoder` (e.g., `Female` $\rightarrow 0$, `Male` $\rightarrow 1$)[cite: 1, 2].
  * **`Geography` Column:** Encoded using `sklearn.preprocessing.OneHotEncoder(drop='first', sparse_output=True)` to handle multi-class geographic features (`France`, `Germany`, `Spain`) while avoiding multi-collinearity[cite: 1, 2].
* **Feature Concat & Cleanup:** Merged One-Hot encoded geographical columns (`Geography_Germany`, `Geography_Spain`) back into the main DataFrame and dropped the original `Geography` feature[cite: 1, 2].

### 2. Preprocessor Persistence
Saved trained preprocessor objects to disk using `pickle` for deployment reproducibility[cite: 1, 2]:
* `labelencoder.pkl`: Label encoder mapping for `Gender`[cite: 1, 2].
* `OHE_Geography.pkl`: One-Hot encoder mapping for `Geography`[cite: 1, 2].
* `scaler.pkl`: Standard scaler object fitted on training features[cite: 1, 2].

### 3. Data Splitting & Feature Scaling
* **Dataset Splitting:** Partitioned dataset into feature set `X` and target `y` (`Exited`), followed by a 67:33 train-test split (`train_test_split(test_size=0.33, random_state=43)`)[cite: 1, 2].
* **Standardization:** Applied `StandardScaler` on input features (`x_train_sc`, `x_test_sc`) to scale data to zero mean and unit variance[cite: 1, 2].

### 4. Neural Network Architecture & Training (Keras / TensorFlow)
Constructed a Sequential Artificial Neural Network (ANN)[cite: 1, 2]:
* **Input Layer / Hidden Layer 1:** `Dense(64 units, activation='relu')`[cite: 1, 2]
* **Hidden Layer 2:** `Dense(32 units, activation='relu')`[cite: 1, 2]
* **Output Layer:** `Dense(1 unit, activation='sigmoid')` for binary classification[cite: 1, 2].
* **Optimization & Loss:**
  * **Optimizer:** `Adam(learning_rate=0.01)`[cite: 1, 2]
  * **Loss Function:** `binary_crossentropy`[cite: 1, 2]
  * **Metrics:** `accuracy`[cite: 1, 2]
* **Callbacks:**
  * **`TensorBoard` Callback:** Logged training metrics into a time-stamped directory (`logs/fit...`)[cite: 1, 2].
  * **`EarlyStopping` Callback:** Monitored `val_loss` with a patience of 15 epochs and restored the best model weights to prevent overfitting[cite: 1, 2].
* **Model Persistence:** Saved trained network to disk as `model.h5`[cite: 1, 2].

### 5. Prediction Pipeline Testing
Loaded persisted artifacts (`model.h5`, `labelencoder.pkl`, `OHE_Geography.pkl`, `scaler.pkl`) to test real-time predictions on sample dictionary inputs[cite: 1, 2].

---

## 📁 Repository Structure

```text
├── Churn_Modelling.csv     # Raw dataset
├── experiments.ipynb       # Data processing, model training & testing notebook
├── app.py                  # Streamlit web interface
├── model.h5                # Trained TensorFlow model
├── labelencoder.pkl        # Pickle file for Gender LabelEncoder
├── OHE_Geography.pkl       # Pickle file for Geography OneHotEncoder
├── scaler.pkl              # Pickle file for StandardScaler
├── requirements.txt        # Dependencies
└── README.md               # Project documentation