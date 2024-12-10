
# **Aerospace Engineering Report: Predicting Remaining Useful Life (RUL) for Engines**
### *Affiliation: Rocket Science Center*

---

## **1. Introduction**

### **1.1 Motivation**
In the aerospace industry, the failure of a critical engine component can have catastrophic consequences, both financially and operationally. **Predictive maintenance** enables engineers to anticipate component failures and schedule maintenance **before failures occur**, reducing unplanned downtimes, repair costs, and improving safety.

Given my work at the **Rocket Science Center**, this project was motivated by the desire to develop an **AI-based predictive maintenance system**. Using machine learning, we aim to predict the **Remaining Useful Life (RUL)** of engines based on historical sensor data.

### **1.2 Problem Statement**
The **Remaining Useful Life (RUL)** is defined as the number of operational cycles an engine can run before it reaches the point of failure. Predicting RUL allows engineers to:

- Optimize maintenance schedules.
- Reduce risks of mid-mission failures.
- Extend asset life by operating components to their limits safely.

The challenge lies in analyzing **multivariate sensor data** and identifying patterns to make accurate predictions.

### **1.3 Objective**
This project aims to:
1. Predict the **Remaining Useful Life (RUL)** of engines using historical sensor and operational setting data.
2. Build and evaluate two machine learning models:
   - **Random Forest Regressor** (Ensemble Decision Trees).
   - **XGBoost Regressor** (Boosted Gradient Decision Trees).
3. Identify which **features (sensors)** contribute most significantly to RUL predictions.

---

## **2. Data Overview**

### **2.1 Dataset Description**
The data consists of three main datasets:
- **Training Data (`PM_train.csv`)**: Historical engine data, including sensor readings, operational settings, and true RUL values.
- **Test Data (`PM_test.csv`)**: Engine data without RUL values (to be predicted).
- **Truth Data (`PM_truth.csv`)**: True RUL values for engines in the test set (used for evaluation).

### **2.2 Data Structure**
The datasets contain the following information:

- **`id`**: Engine ID.
- **`cycle`**: Operational cycle number for the engine.
- **`setting1`, `setting2`, `setting3`**: Operational settings for the engine.
- **`s1` to `s21`**: Sensor measurements capturing engine performance.
- **`RUL`**: Remaining Useful Life (only available in training data).

| id | cycle | setting1 | setting2 | setting3 | s1   | s2   | ... | s21   | RUL |
|----|-------|----------|----------|----------|------|------|-----|-------|-----|
| 1  | 1     | -0.0007  | -0.0004  | 100.0    | 518.67 | 641.82 | ... | 23.4190 | 191 |
| 1  | 2     | 0.0019   | -0.0003  | 100.0    | 518.67 | 642.15 | ... | 23.4236 | 190 |

### **2.3 Terminology**
- **RUL (Remaining Useful Life)**: The number of cycles left before engine failure.
- **Cycle**: A single operational phase or period for an engine.
- **Features**:
   - **Operational Settings**: Parameters that dictate engine operation (e.g., load, speed).
   - **Sensors**: Measurements of engine parameters like temperature, pressure, vibration, etc.

---

## **3. Methodology**

### **3.1 Data Preprocessing**
#### **Step 1: Feature and Target Separation**
To train machine learning models, the data must be split into:
- **Features (`X`)**: Predictor variables (`setting1`, `s1` to `s21`).
- **Target (`y`)**: Remaining Useful Life (RUL).

```python
features_to_drop = ["id", "cycle", "RUL"]
X = train.drop(features_to_drop, axis=1)
y = train["RUL"]
```

- **Why Drop `id` and `cycle`?**
   - `id` is an identifier with no predictive value.
   - `cycle` is implicitly accounted for in the target (RUL).

#### **Step 2: Train-Validation Split**
To evaluate model performance, the data is split into **training** and **validation** sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **80-20 Split**: 80% of data is used for training, 20% for validation.
- **`random_state=42`**: Ensures reproducibility.

---

### **3.2 Model Selection and Training**
#### **3.2.1 Random Forest Regressor**
The Random Forest model is an ensemble of decision trees where:
- Multiple trees are built using random subsets of data (bagging).
- Final predictions are averaged to reduce variance.

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

- **`n_estimators=100`**: Number of decision trees.
- **`random_state=42`**: Ensures consistent results.

**Model Performance**:
```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred_rf = rf_model.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
print(f"RandomForest RMSE: {rmse_rf}")
```

- **Root Mean Squared Error (RMSE)**: Measures prediction accuracy. Lower RMSE is better.

---

#### **3.2.2 XGBoost Regressor**
XGBoost is a **boosted gradient decision tree** model that:
- Corrects errors made by previous trees iteratively.
- Optimizes both speed and accuracy.

```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, objective="reg:squarederror")
xgb_model.fit(X_train, y_train)
```

- **`learning_rate=0.05`**: Step size for each iteration.
- **`n_estimators=200`**: Number of boosting rounds.

**Performance**:
```python
y_pred_xgb = xgb_model.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb}")
```

---

## **4. Results**

### **4.1 Model Performance**
| Model                 | RMSE (Validation) |
|------------------------|-------------------|
| Random Forest          | **41.52**        |
| XGBoost                | **41.35**        |

- **Observation**: XGBoost performed slightly better.

---

### **4.2 Feature Importance**
Random Forest's feature importance plot reveals which sensors contribute most to RUL predictions.

![Feature Importance](Screenshot)

- **Top Features**:
   - `s11`: Dominant sensor.
   - `s9`, `s12`, and `s4`: Other significant contributors.
   - Operational settings (`setting1`, `setting2`, `setting3`) have minimal impact.

---

## **5. Conclusion**

### **Findings**
1. **XGBoost** achieved the best performance with an RMSE of **41.35**.
2. Sensor **`s11`** is the most critical predictor of Remaining Useful Life.
3. Predictive maintenance models can optimize engine performance and safety.

### **Recommendations**
1. Prioritize monitoring **sensor `s11`** for maintenance systems.
2. Deploy the XGBoost model for real-time RUL predictions.
3. Extend this approach to other critical components in aerospace systems.

---

## **6. Future Work**
1. Implement **LSTM (Long Short-Term Memory)** models for time-series predictions.
2. Integrate the model into a **real-time IoT-based monitoring system**.
3. Fine-tune models further using larger datasets.

---

**End of Report** 🚀
