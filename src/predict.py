import pandas as pd
import joblib

# Load model
model = joblib.load('model/model.pkl')

# Sample patient
sample = pd.DataFrame({
   '5.1':[5.0],
   '3.5':[3.0],
   '1.4':[1.4],
   '0.2':[0.2],
})

prediction = model.predict(sample)[0]
print(f" Prediction: {prediction}")
