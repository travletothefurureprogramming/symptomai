import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

sample = pd.DataFrame([{
    "fever": 1,
    "cough": 1,
    "headache": 1,
    "fatigue": 1,
    "sore_throat": 1,
    "runny_nose": 1,
    "nausea": 0,
    "vomiting": 0,
    "diarrhea": 0,
    "muscle_pain": 0,
    "sneezing": 0,
    "difficulty_breathing": 0
}])

prediction = model.predict(sample)

print(f"Prediction: {prediction[0]}")