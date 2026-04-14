
# Machine Learning Based prediction of Pulmonary Cancer Prediction using Clinical Survey Data

## 📌 Project Overview

This project focuses on analyzing a lung cancer dataset and building a machine learning model to predict whether a person has lung cancer based on various health and lifestyle factors.

The project includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Data visualization
* Handling imbalanced data
* Model building using Logistic Regression
* Model evaluation

---

## 📂 Dataset Information

The dataset used in this project is **"survey lung cancer.csv"**, which contains information about patients such as:

* Gender
* Age
* Smoking habits
* Anxiety
* Peer pressure
* Chronic disease
* Fatigue
* Allergy
* Wheezing
* Alcohol consumption
* Coughing
* Shortness of breath
* Swallowing difficulty
* Chest pain
* Lung cancer (Target variable)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🔍 Project Workflow

### 1. Data Loading

* Loaded dataset using Pandas
* Checked dataset structure, columns, and data types

### 2. Data Cleaning

* Checked for missing values (no null values found)
* Verified unique values in each column

### 3. Exploratory Data Analysis (EDA)

* Univariate Analysis:

  * Gender distribution
  * Age distribution
  * Smoking status
  * Other health factors
* Visualizations:

  * Bar charts
  * Histograms
  * Pie charts

### 4. Multivariate Analysis

* Used Seaborn count plots to analyze relationships between features and lung cancer
* Boxplot used to analyze age vs cancer

### 5. Data Preprocessing

* Encoded categorical variables
* Handled class imbalance in dataset

### 6. Train-Test Split

* Split dataset into training and testing sets (80:20 ratio)

### 7. Model Building

* Used **Logistic Regression** for classification

### 8. Model Evaluation

* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

## 📊 Results

* The model was able to predict lung cancer with good accuracy
* Confusion matrix helps visualize prediction performance
* Classification report shows precision, recall, and F1-score

---

## ▶️ How to Run the Project

1. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Place dataset file:

```
survey lung cancer.csv
```

3. Run the Jupyter Notebook:

```
Final Pulmonary Cancer.ipynb
```

---

## 📈 Future Improvements

* Try advanced models (Random Forest, SVM, XGBoost)
* Improve accuracy with hyperparameter tuning
* Deploy as a web app using Flask or Streamlit
* Add real-time prediction interface

---

## 👨‍💻 Author

**S. Chaitanya**

* B.Tech CSE (Data Science)
* Skills: Python, Data Science, Web Development

---

## ⭐ Conclusion

This project demonstrates how machine learning can help in early detection of lung cancer using patient data. It showcases the complete pipeline from data analysis to model evaluation.

---
