# Project: Predicting Credit Card Default Risk
# Authors: Erik Redman, Jesse Poljak, Marwen Boughanmi, and Esha Sharma

## Executive Summary
This project analyzes and predicts the probability of customer default by leveraging a dataset containing demographic and financial features. By conducting thorough data preprocessing, building optimized machine learning models, and evaluating their performance with key metrics, this project supports better risk management decisions. The insights enable proactive credit risk management for lenders.

---

## Key Steps

### 1. Data Preprocessing
- Cleaned and transformed the dataset to ensure accuracy and consistency.
- Addressed class imbalance using oversampling techniques like **SMOTE**.

### 2. Model Development
- Built and tested multiple machine learning models.
- Selected **Random Forest Classifier** as the final model.
- Optimized hyperparameters using **RandomizedSearchCV**.

### 3. Performance Evaluation
- Evaluated model effectiveness with metrics such as:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**

---

## Motivation

### Risk Mitigation
- Identifies high-risk credit card holders, allowing for early intervention.

### Improved Decision-Making
- Supports lenders in making informed credit limit decisions.

---

## Model Evaluation Attempts:

### 1. Balanced Random Forest Classifier

#### Objective:
To classify data using a Balanced Random Forest Classifier, which is designed to handle class imbalance by using bootstrapping and balancing the classes during the training process.

#### Code:
```python
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report

# Initialize and fit the Balanced Random Forest Classifier
clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
```

#### Results:
The model achieved a moderate performance with an accuracy of 74%, but the recall for the minority class (1) was low. This suggests potential issues with class imbalance, despite the model's adjustments for balancing during training.

### 2. Logistic Regression

#### Objective:
To evaluate the performance of a Logistic Regression model on both unscaled and scaled data (Standard Scaling and Min-Max Scaling).

#### 1. Unscaled Data

##### Code:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)

# Print accuracy scores
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred_test))
```
#####  Results:
Training Accuracy: 77.8%
Testing Accuracy: 78.08%

#### 2. Standard Scaling
##### Code:
```python
from sklearn.preprocessing import StandardScaler

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Predict and evaluate accuracy
y_pred_train_scaled = logreg.predict(X_train_scaled)
y_pred_test_scaled = logreg.predict(X_test_scaled)

# Print accuracy scores
print("Training Accuracy (Standard Scaling):", accuracy_score(y_train, y_pred_train_scaled))
print("Testing Accuracy (Standard Scaling):", accuracy_score(y_test, y_pred_test_scaled))
```

#### Results:
Training Accuracy: 80.84%
Testing Accuracy: 81.45%
Improved performance with scaling.

#### 3. Min-Max Scaling
##### Code:
``` python
from sklearn.preprocessing import MinMaxScaler

# Min-Max Scaling
scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(X_train)
X_test_minmax = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_minmax, y_train)

# Predict and evaluate accuracy
y_pred_train_minmax = logreg.predict(X_train_minmax)
y_pred_test_minmax = logreg.predict(X_test_minmax)

# Print accuracy scores
print("Training Accuracy (Min-Max Scaling):", accuracy_score(y_train, y_pred_train_minmax))
print("Testing Accuracy (Min-Max Scaling):", accuracy_score(y_test, y_pred_test_minmax))
```

##### Results:
Training Accuracy: 80.82%
Testing Accuracy: 81.35%

### 3. K-Nearest Neighbors (KNN)

#### Objective:
To evaluate the performance of a K-Nearest Neighbors (KNN) model on both unscaled and scaled data, testing for different values of k.

#### 1. Unscaled Data

##### Code:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test KNN for various k values (1 to 19)
best_k = 1
best_accuracy = 0

for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best k: {best_k} with accuracy: {best_accuracy}")
```
##### Results:
Peak performance observed at k=19 with an accuracy of 77.9%.

#### 2. Standard Scaled Data
##### Code:
``` python

from sklearn.preprocessing import StandardScaler

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test KNN for various k values (1 to 19)
best_k = 1
best_accuracy = 0

for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best k: {best_k} with accuracy: {best_accuracy}")
```
##### Results:
Scaling significantly improved results.
Optimal k=13, with Test Accuracy: 81.5%.

### 4. Support Vector Machine (SVM)
#### Objective: 
To evaluate the performance of a Support Vector Machine (SVM) with a Polynomial kernel on scaled data.

#### Configuration:
Kernel: Polynomial (poly)
Scaler: Standard Scaler
#### Code:
``` python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM model with a polynomial kernel
svm = SVC(kernel='poly')
svm.fit(X_train_scaled, y_train)

# Predict and evaluate accuracy
y_pred_train = svm.predict(X_train_scaled)
y_pred_test = svm.predict(X_test_scaled)

# Print accuracy scores
print("Training Accuracy (SVM):", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy (SVM):", accuracy_score(y_test, y_pred_test))
```
#### Results:
Train Accuracy: 81.2%
Test Accuracy: 81.1%

### 5. Random Forest Classifier
#### Configuration:
n_estimators: 100
max_depth: 6

#### Code:
``` python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Initialize and train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate performance
y_pred = rf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
``` 
#### Results:
``` python
             precision    recall  f1-score   support

           0       0.85      0.91      0.88
           1       0.67      0.50      0.57

    accuracy                           0.81  
```
### Conclusion: 
The **Random Forest Classifier Search CV with Hyperparameter Tuning** was chosen as the final model for its accurate performance and precision results. The model can handle imbalanced data and high-dimensional features making it ideal for credit card default prediction.

# Conclusion: 
Our model seems to be not having great power predicting the defaut customer, however its a lot better predicting not default customer.
Predicting credit card default is a critical task for financial institutions to mitigate risk and optimize their lending strategies. By leveraging advanced machine learning techniques and robust data analysis, it is possible to build accurate models that can identify potential defaulters early on.

