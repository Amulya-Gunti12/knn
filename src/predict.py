import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load data
df = pd.read_csv('data/iris.data')
X = df.drop('Iris-setosa', axis=1)
y = df['Iris-setosa']

# Train model
model = KNeighborsClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model/model.pkl')
print(" iris model trained.")
