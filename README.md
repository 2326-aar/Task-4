# Task-4
# 🧠 Logistic Regression - Binary Classification Task

## 📌 Task Overview
This project is part of an AI & ML internship task focusing on binary classification using **Logistic Regression**. The objective is to train a classifier to distinguish between malignant and benign breast tumors using the **Breast Cancer Wisconsin Dataset**.

---

## 🔧 Tools & Libraries Used
- Python 🐍
- `pandas` – data handling
- `scikit-learn` – model building and evaluation
- `matplotlib`, `seaborn` – visualization

---

## 📊 Dataset Information
**Dataset**: [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
- **Features**: 30 numeric features (e.g., radius, texture, smoothness)  
- **Target**:  
  - `0` → Malignant (cancerous)  
  - `1` → Benign (non-cancerous)

---

## 🚀 Steps Performed

1. **Data Loading**  
   Used `sklearn.datasets.load_breast_cancer()` to load the data.

2. **Preprocessing**  
   - Train-test split (80/20)  
   - Standardization using `StandardScaler`

3. **Model Training**  
   Trained a `LogisticRegression` model on the training data.

4. **Evaluation Metrics**
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-score)  
   - ROC Curve and ROC-AUC Score  
   
---

## ✅ Results

- **Confusion Matrix**: 
  - True Positives and Negatives clearly highlighted
- **ROC-AUC Score**: ~0.99
- **Model performed very well in separating malignant vs benign cases.**


## 📈 Visuals

### Confusion Matrix  
![Confusion Matrix](screenshots/confusion_matrix.png)

### ROC Curve  
![ROC Curve](screenshots/roc_curve.png)

