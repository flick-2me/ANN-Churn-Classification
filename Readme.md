# Customer Churn Prediction & Salary Regression using Artificial Neural Networks (ANN)

This project builds an end-to-end Machine Learning and Deep Learning pipeline using TensorFlow/Keras and Streamlit. It encompasses binary classification for customer churn, continuous regression for salary prediction, and comprehensive model optimization using GridSearchCV.

🔗 **Live Streamlit App (Churn Classifier):** [https://ann-churn-classification-vw99gjqh994mvh7mvgrwjn.streamlit.app/](https://ann-churn-classification-vw99gjqh994mvh7mvgrwjn.streamlit.app/)

> **Note:** Only the Customer Churn Classification pipeline is deployed on the interactive Streamlit web application. The Salary Regression model and hyperparameter tuning experiments are documented and retained within the repository notebooks.

---

## 📌 Project Overview

Customer retention and behavioral financial forecasting are critical for banking operations. This repository contains complete workflows for:
1. **Churn Classification:** Predicting whether a bank customer will exit (`Exited = 1`) or stay (`Exited = 0`).
2. **Salary Regression:** Estimating a customer's continuous income (`EstimatedSalary`).
3. **Architecture Optimization:** Systematically finding the best neural network parameters for maximum accuracy and minimal loss.

---

## 🛠️ 1. Churn Classification Workflow (`experiments.ipynb`)

The foundational notebook documents the experimental workflow, from raw data preprocessing to deep learning model execution:

### Data Cleaning & Feature Engineering
* **Data Ingestion:** Loaded customer records from `Churn_Modelling.csv` using `pandas`.
* **Identifier Removal:** Dropped non-predictive columns (`RowNumber`, `CustomerId`, `Surname`).
* **Categorical Encoding:**
  * **`Gender` Column:** Encoded using `sklearn.preprocessing.LabelEncoder` (e.g., `Female` $\rightarrow 0$, `Male` $\rightarrow 1$).
  * **`Geography` Column:** Encoded using `sklearn.preprocessing.OneHotEncoder(drop='first', sparse_output=True)` to handle multi-class geographic features (`France`, `Germany`, `Spain`) while avoiding multi-collinearity.
* **Feature Concat & Cleanup:** Merged One-Hot encoded geographical columns (`Geography_Germany`, `Geography_Spain`) back into the main DataFrame and dropped the original `Geography` feature.

### Preprocessor Persistence
Saved trained preprocessor objects to disk using `pickle` for deployment reproducibility:
* `labelencoder.pkl`: Label encoder mapping for `Gender`.
* `OHE_Geography.pkl`: One-Hot encoder mapping for `Geography`.
* `scaler.pkl`: Standard scaler object fitted on training features.

### Data Splitting & Feature Scaling
* **Dataset Splitting:** Partitioned dataset into feature set `X` and target `y` (`Exited`), followed by a 67:33 train-test split.
* **Standardization:** Applied `StandardScaler` on input features to scale data to zero mean and unit variance.

### Neural Network Architecture & Training (Classification)
Constructed a Sequential Artificial Neural Network (ANN):
* **Input Layer / Hidden Layer 1:** `Dense(64 units, activation='relu')`
* **Hidden Layer 2:** `Dense(32 units, activation='relu')`
* **Output Layer:** `Dense(1 unit, activation='sigmoid')` for binary classification probabilities.
* **Compilation & Callbacks:**
  * **Optimizer/Loss:** `Adam(learning_rate=0.01)` / `binary_crossentropy`.
  * **`TensorBoard` Callback:** Logged training metrics into a time-stamped directory (`logs/fit...`).
  * **`EarlyStopping` Callback:** Monitored `val_loss` with a patience of 15 epochs.
* **Model Persistence:** Saved trained network to disk as `model.h5`.

---

## 📈 2. Salary Regression Workflow (`Salary_regression.ipynb`)

This phase adapts the deep learning architecture to predict continuous numerical values instead of categorical classes:
* **Target Variable Shift:** The target `y` was changed from the binary `Exited` column to the continuous `EstimatedSalary` column.
* **Regression Output Layer:** Replaced the final layer with a single neuron and no activation function (e.g., `Dense(1, activation='linear')`) to allow the network to output unbounded continuous salary values.
* **Loss Functions:** Replaced binary cross-entropy with regression-specific loss metrics like Mean Squared Error (`MSE`) and Mean Absolute Error (`MAE`).
* **Persistence:** The final trained regression weights were saved locally as `regression_model.h5`.

---

## ⚙️ 3. Hyperparameter Tuning (`hyperparameter_tuning.ipynb`)

To move past trial-and-error, the network architecture was programmatically optimized to find the most efficient parameters:
* **SciKeras Integration:** Bridged TensorFlow/Keras with Scikit-Learn using `scikeras.wrappers.KerasClassifier`.
* **GridSearchCV Implementation:** Automated cross-validation over a predefined dictionary (`param_grid`) of network constraints, executing on a single thread (`n_jobs=1`) to prevent process pickling errors.
* **Tuning Parameters Explored:**
  * **Network Depth:** Iterated between models with single vs. multiple hidden layers.
  * **Neurons per Layer:** Tested varying capacities (e.g., 16, 32, 64 nodes) to find the threshold between underfitting and overfitting.
  * **Optimization Algorithms:** Compared the convergence speed and accuracy of `adam` vs. `rmsprop`.
  * **Training Configurations:** Evaluated combinations of `batch_size` (e.g., 16, 32) and `epochs` (e.g., 10, 20).

---

## 💻 4. Interactive Web Deployment (`app.py`)

A user-friendly web interface built with **Streamlit**:
* Restores the persisted `model.h5` and preprocessor `.pkl` files.
* Accepts manual inputs for Credit Score, Geography, Gender, Age, Balance, etc.
* Processes inputs through the exact same scaling pipeline used during training.
* Outputs a real-time risk assessment indicating whether a customer is likely to churn.

---

## 📁 Repository Structure

```text
├── Churn_Modelling.csv         # Raw dataset
├── experiments.ipynb           # Churn classification workflow & baseline ANN
├── Salary_regression.ipynb     # Salary regression pipeline & experiments
├── hyperparameter_tuning.ipynb # GridSearchCV architecture optimization
├── app.py                      # Streamlit web interface (Churn Classifier)
├── model.h5                    # Trained TensorFlow Classification model
├── regression_model.h5         # Trained TensorFlow Regression model
├── labelencoder.pkl            # Pickle file for Gender LabelEncoder
├── OHE_Geography.pkl           # Pickle file for Geography OneHotEncoder
├── scaler.pkl                  # Pickle file for StandardScaler
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
