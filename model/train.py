from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

# Charger le dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline : prétraitement + modèle
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Faire une prédiction sur le jeu de test
y_pred = pipeline.predict(X_test)

# Calculer la métrique
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Créer le dossier artifacts si besoin
os.makedirs("artifacts", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(pipeline, "artifacts/model.pkl")
print("Model saved to artifacts/model.pkl")