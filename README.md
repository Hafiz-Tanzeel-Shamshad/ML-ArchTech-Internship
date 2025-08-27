# 🔍 Machine Learning Projects – Internship Tasks

This repository contains solutions to two machine learning tasks:

1. **📧 Email Spam Classification using Naive Bayes**
2. **🧠 Handwritten Digit Classification using a Fully Connected Neural Network (FCNN)**
3. **🌸Iris Flower Classification**
4. **🏡California Housing Price Prediction** 
---

# 📧 Task 1: Email Spam Classification using Naive Bayes

This project builds a machine learning model to classify email or SMS messages as **spam** or **not spam** using Natural Language Processing (NLP) and the **Multinomial Naive Bayes** algorithm.

---

## 🚀 Overview

The objective is to:
- Preprocess raw text (lowercasing, stopword removal, stemming)
- Extract features using **TF-IDF**
- Train and evaluate a Naive Bayes model
- Output performance metrics (accuracy, precision, recall, F1-score)

---

## 🧰 Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Multinomial Naive Bayes Classifier

---

## 📁 Dataset

We use the **UCI SMS Spam Collection Dataset**, which includes 5,574 SMS messages labeled as either `ham` or `spam`.

📥 Dataset Source:  
[UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

```python
# Example to load data
import pandas as pd
data = pd.read_csv(
    'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
    sep='\t', names=["label", "message"]
)

```
## 🧪 Model Pipeline

### 📝 Text Preprocessing
- ✅ Lowercase conversion  
- ✅ Punctuation removal  
- ✅ Stopword removal *(using NLTK)*  
- ✅ Stemming *(using PorterStemmer)*  

### 📊 Feature Extraction
- ✅ TF-IDF vectorization  

### 🧠 Model Training
- ✅ Train/Test Split: **80/20**  
- ✅ Model Used: **Multinomial Naive Bayes**  

### 📈 Evaluation Metrics
- ✅ Accuracy  
- ✅ Confusion Matrix  
- ✅ Precision  
- ✅ Recall  
- ✅ F1-score


### 📈 Evaluation Metrics

- **Accuracy**: `98.5%`

#### 📋 Classification Report

| Label | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| **ham**  | 0.99      | 1.00   | 0.99     | 965     |
| **spam** | 1.00      | 0.90   | 0.95     | 150     |

# 🧠 Task 2: MNIST Digit Classification using FCNN

This project demonstrates a basic deep learning workflow using a **Fully Connected Neural Network (FCNN)** to classify handwritten digits from the **MNIST** dataset.

---

## 🧾 Objective

- 📥 Load and preprocess the MNIST dataset  
- 🏗️ Build a simple FCNN using TensorFlow/Keras  
- 🎯 Train the model using backpropagation  
- 📈 Evaluate using classification metrics and visualizations  

---

## 📦 Dataset: MNIST

- 📊 60,000 training images  
- 📊 10,000 testing images  
- 🖼️ Grayscale 28x28 images  
- 🔢 Classes: Digits `0–9`  

---

## 🛠️ Tech Stack

- 🐍 Python 3.x  
- 🔧 TensorFlow / Keras  
- 📐 NumPy  
- 📊 Matplotlib  

---

## 🧠 Model Architecture

| Layer Type      | Output Shape     | Activation     |
|-----------------|------------------|----------------|
| Flatten         | (784,)           | -              |
| Dense (128)     | (128,)           | LeakyReLU (α=0.1) |
| Dense (10)      | (10,)            | Softmax        |

- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  
- **Epochs**: 5  
- **Batch Size**: 32  
- **Validation Split**: 20% of training data  

---

## 📈 Results

- ✅ **Test Accuracy**: ~98%  
- 🎯 Visualizations show correct predictions on test samples.

---

## 📊 Sample Output

```python
Training data shape: (60000, 28, 28)
Test data shape: (10000, 28, 28)
Test Accuracy: 0.9803
```
# 🌸 Task 3: Iris Flower Classification 
- **Objective**: Classify iris flowers into three species (`setosa`, `versicolor`, `virginica`) based on sepal length/width and petal length/width.
- **Dataset**: [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) (available in `sklearn`).
- **Algorithm Used**: Logistic Regression
- **Steps**:
  - Load and explore dataset  
  - Preprocess features and target  
  - Train/test split (80/20)  
  - Train Logistic Regression model  
  - Evaluate with accuracy & classification report  
  - Make predictions on new flower samples  

✅ Achieved **100% accuracy** on test data.  

---

# 🏡 Task 4: California Housing Price Prediction 
- **Objective**: Predict median house values in California districts using features such as median income, average rooms, population, latitude, and longitude.
- **Dataset**: [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (available in `sklearn`).
- **Algorithm Used**: Linear Regression
- **Steps**:
  - Load dataset  
  - Data cleaning (check missing values)  
  - Select important features  
  - Train/test split (80/20)  
  - Train Linear Regression model  
  - Evaluate with Mean Squared Error (MSE) and R² score  
  - Compare predicted vs. actual prices  

✅ Achieved **~57% R² Score** (baseline model).  
## 📜 License

This project is open-source and available under the **MIT License**.

