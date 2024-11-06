import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv('./Wines.csv')

X = df.drop(columns=["quality"]) 
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Initialisation des modèles
models = {
    "Random Forest": RandomForestClassifier(),
}

# Entraîner chaque modèle et évaluer ses performances
for name, model in models.items():
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Afficher le rapport de classification pour plus de détails
    print(f"\nClassification Report for {name}:\n", classification_report(y_test, y_pred))
    print("-" * 50)
    
with open("wine_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé dans 'wine_quality_model.pkl'")
