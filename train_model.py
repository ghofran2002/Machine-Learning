import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Charger les données
df = pd.read_csv('music.csv')

# Préparer les features et la cible
X = df.drop(columns=['genre'])
y = df['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Évaluer le modèle
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Sauvegarder le modèle
joblib.dump(model, 'music_recommender.joblib')
