# Task-4
# 🧠 Logistic Regression Classifier - Breast Cancer Detection

## 📌 Project Summary
This project uses logistic regression for binary classification of breast cancer using the `data task 4.csv` dataset. The aim is to classify tumors as **malignant (1)** or **benign (0)** based on numerical features extracted from digitized images.

---

## 📊 Dataset Overview
- **File**: `data task 4.csv`
- **Source**: Breast Cancer Wisconsin Dataset
- **Target Column**: `diagnosis` (`M` = malignant, `B` = benign → mapped to 1 and 0)
- **Features**: 30 numeric columns describing tumor characteristics

---

## 🛠 Tools and Libraries
- Python 3
- `pandas`, `numpy` for data processing
- `matplotlib`, `seaborn` for plotting
- `scikit-learn` for machine learning and evaluation

---

## 🚀 Workflow

1. **Data Cleaning**
   - Dropped unnecessary columns (`id`, `Unnamed: 32`)
   - Converted diagnosis to binary labels (M → 1, B → 0)

2. **Preprocessing**
   - Train-test split (80% train, 20% test)
   - Feature standardization with `StandardScaler`

3. **Model Training**
   - Applied `LogisticRegression` with increased iterations

4. **Evaluation**
   - Confusion Matrix
   - Precision, Recall, F1-Score (Classification Report)
   - ROC Curve and AUC Score
   - Sigmoid function visualization
   
---

## ✅ Results

- **Model**: Logistic Regression
- **Accuracy**: High (over 95% expected)
- **ROC-AUC Score**: Typically > 0.98
- **Clear separation between malignant and benign classes**

---

## 📈 Visuals

- 📊 Confusion Matrix  
- 📈 ROC Curve  
- ➿ Sigmoid Curve  

