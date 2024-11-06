import pickle
import numpy as np

# Charger le modèle entraîné depuis un fichier (par exemple, "wine_quality_model.pkl")
with open("wine_quality_model.pkl", "rb") as f:
    model = pickle.load(f)

# Définir une fonction pour effectuer la prédiction
def predict_wine_quality(data):
    # Extraire les caractéristiques sous forme de liste ou de tableau numpy
    features = np.array([[
        data["fixed_acidity"],
        data["volatile_acidity"],
        data["citric_acid"],
        data["residual_sugar"],
        data["chlorides"],
        data["free_sulfur_dioxide"],
        data["total_sulfur_dioxide"],
        data["density"],
        data["pH"],
        data["sulphates"],
        data["alcohol"],
    ]])

    # Utiliser le modèle pour prédire la qualité
    prediction = model.predict(features)
    
    return int(prediction[0])  