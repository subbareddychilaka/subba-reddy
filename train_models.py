# ============================================
# MULTI-DISEASE MODEL TRAINING SCRIPT
# Diabetes | Heart | Liver | Kidney
# Federated Learning + XGBoost Ensemble
# ============================================

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# CREATE MODELS DIRECTORY
# ============================================
os.makedirs("models", exist_ok=True)

# ============================================
# FEDERATED NEURAL NETWORK
# ============================================
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def federated_train(X, y, rounds=15):
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clients = np.array_split(np.column_stack((X_train, y_train)), 5)
    global_model = create_model(X_train.shape[1])

    for _ in range(rounds):
        weights, sizes = [], []

        for client in clients:
            X_c, y_c = client[:, :-1], client[:, -1]
            local = create_model(X_train.shape[1])
            local.set_weights(global_model.get_weights())
            local.fit(X_c, y_c, epochs=5, batch_size=16, verbose=0)
            weights.append(local.get_weights())
            sizes.append(len(X_c))

        new_weights = []
        for layer in zip(*weights):
            new_weights.append(np.average(layer, axis=0, weights=sizes))

        global_model.set_weights(new_weights)

    return global_model, scaler, X_train, y_train


# ============================================
# DIABETES
# ============================================
print("\nTraining Diabetes Model...")
diabetes = pd.read_csv("diabetes.csv")
X_d = diabetes.drop("Outcome", axis=1)
y_d = diabetes["Outcome"]

model_d, scaler_d, Xtr_d, ytr_d = federated_train(X_d, y_d)
xgb_d = XGBClassifier(n_estimators=200)
xgb_d.fit(Xtr_d, ytr_d)

joblib.dump(model_d, "models/diabetes_nn.pkl")
joblib.dump(xgb_d, "models/diabetes_xgb.pkl")
joblib.dump(scaler_d, "models/diabetes_scaler.pkl")
joblib.dump(X_d.columns.tolist(), "models/diabetes_columns.pkl")


# ============================================
# HEART
# ============================================
print("Training Heart Model...")
heart = pd.read_csv("heart.csv")
X_h = heart.drop("target", axis=1)
y_h = heart["target"]

X_h, y_h = SMOTE().fit_resample(X_h, y_h)

model_h, scaler_h, Xtr_h, ytr_h = federated_train(X_h, y_h)
xgb_h = XGBClassifier(n_estimators=200)
xgb_h.fit(Xtr_h, ytr_h)

joblib.dump(model_h, "models/heart_nn.pkl")
joblib.dump(xgb_h, "models/heart_xgb.pkl")
joblib.dump(scaler_h, "models/heart_scaler.pkl")
joblib.dump(X_h.columns.tolist(), "models/heart_columns.pkl")


# ============================================
# LIVER
# ============================================
print("Training Liver Model...")
liver = pd.read_csv("liver.csv")

if "Gender" in liver.columns:
    liver["Gender"] = LabelEncoder().fit_transform(liver["Gender"])

liver = liver.fillna(liver.median())

X_l = liver.drop("Dataset", axis=1)
y_l = liver["Dataset"].map({1: 1, 2: 0})

X_l, y_l = SMOTE().fit_resample(X_l, y_l)

model_l, scaler_l, Xtr_l, ytr_l = federated_train(X_l, y_l)
xgb_l = XGBClassifier(n_estimators=200)
xgb_l.fit(Xtr_l, ytr_l)

joblib.dump(model_l, "models/liver_nn.pkl")
joblib.dump(xgb_l, "models/liver_xgb.pkl")
joblib.dump(scaler_l, "models/liver_scaler.pkl")
joblib.dump(X_l.columns.tolist(), "models/liver_columns.pkl")


# ============================================
# KIDNEY
# ============================================
print("Training Kidney Model...")
kidney = pd.read_csv("kidney_disease.csv")

kidney.replace({
    "yes": 1, "no": 0,
    "present": 1, "notpresent": 0,
    "abnormal": 1, "normal": 0,
    "poor": 1, "good": 0,
    "ckd": 1, "notckd": 0
}, inplace=True)

kidney = kidney.apply(pd.to_numeric, errors="coerce")
kidney = kidney.fillna(kidney.median())

X_k = kidney.drop("classification", axis=1)
y_k = kidney["classification"]

X_k, y_k = SMOTE().fit_resample(X_k, y_k)

model_k, scaler_k, Xtr_k, ytr_k = federated_train(X_k, y_k)
xgb_k = XGBClassifier(n_estimators=200)
xgb_k.fit(Xtr_k, ytr_k)

joblib.dump(model_k, "models/kidney_nn.pkl")
joblib.dump(xgb_k, "models/kidney_xgb.pkl")
joblib.dump(scaler_k, "models/kidney_scaler.pkl")
joblib.dump(X_k.columns.tolist(), "models/kidney_columns.pkl")


print("\n✅ ALL MODELS TRAINED & SAVED SUCCESSFULLY")
print("📁 Check the 'models/' folder")
