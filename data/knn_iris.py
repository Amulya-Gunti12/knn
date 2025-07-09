import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load CSV
df = pd.read_csv('iris.data')

# Clean column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Encode target labels (e.g., setosa -> 0)
le = LabelEncoder()
y = le.fit_transform(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
