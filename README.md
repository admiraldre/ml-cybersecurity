# Neural Network Classification Model

## Overview
This project implements a classification model using a neural network to predict attack categories from the **"SENG 4610 Training Data"** dataset. The workflow includes data preprocessing, feature engineering, model training, evaluation, and visualization.

## Features
- Encodes categorical variables into numerical labels.
- Normalizes features using `StandardScaler`.
- Implements a multi-layer neural network for multi-class classification.
- Visualizes model performance using accuracy/loss curves, confusion matrix heatmaps, ROC, and Precision-Recall curves.

## Technologies Used
- **Python**
- **Libraries**: 
  - `pandas` 
  - `numpy` 
  - `scikit-learn` 
  - `tensorflow` 
  - `matplotlib` 
  - `seaborn`

## Dataset
The dataset is expected to be a CSV file named **`SENG 4610 Training Data.csv`**. It contains:
- Features for classification.
- Columns such as `proto`, `service`, `state`, and `attack_cat` to be encoded.
- The `label` column, which is used for target classification.

---

## Workflow

### 1. **Data Preprocessing**
- Drop the `id` column.
- Encode categorical columns (`proto`, `service`, `state`, `attack_cat`) using `LabelEncoder`.
- Normalize the feature set using `StandardScaler`.
- One-hot encode the target labels.

### 2. **Train-Test Split**
The data is split into:
- **Training set**: 80%
- **Testing set**: 20%

### 3. **Model Architecture**
- The neural network is implemented using TensorFlow's Keras with the following architecture:
  - **Input Layer**: Matches the number of features in the dataset.
  - **Hidden Layers**:
    - Layer 1: 128 neurons, ReLU activation, 30% dropout.
    - Layer 2: 64 neurons, ReLU activation, 30% dropout.
  - **Output Layer**: Matches the number of classes, with softmax activation.

### 4. **Training**
- **Optimizer**: Adam.
- **Loss Function**: Categorical Cross-Entropy.
- **Metrics**: Accuracy.
- **Batch size**: 64.
- **Epochs**: 100.
- **Validation split**: 20%.

### 5. **Evaluation and Metrics**
- **Test Accuracy**: Evaluates model performance on unseen data.
- **Classification Report**: Includes precision, recall, and F1-score for each class.
- **Confusion Matrix**: Displays predictions versus actual values in a heatmap.
- **ROC Curve**: Plots sensitivity and specificity for each class.
- **Precision-Recall Curve**: Displays precision and recall metrics for all classes.

### 6. **Model Saving**
The trained model is saved as **`finalmodel.keras`** for future reuse.

---

## Visualizations
- **Accuracy and Loss Curves**: Tracks training and validation performance over epochs.
- **Confusion Matrix Heatmap**: Visualizes classification performance per class.
- **ROC Curves**: Displays the performance of each class with AUC values.
- **Precision-Recall Curves**: Highlights precision and recall metrics for all classes.

---

## Usage

### Prerequisites
1. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn