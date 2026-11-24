
import customtkinter as ctk
import pandas as pd
import joblib

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("SymptomAI")
app.geometry("500x720")
app.resizable(False, False)

model = joblib.load("models/model.pkl")

main_frame = ctk.CTkFrame(app, corner_radius=20)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

title_label = ctk.CTkLabel(
    main_frame,
    text="SymptomAI",
    font=("Arial", 32, "bold")
)
title_label.pack(pady=(20, 5))

subtitle = ctk.CTkLabel(
    main_frame,
    text="Machine Learning Classification Demo",
    font=("Arial", 13),
    text_color="gray"
)
subtitle.pack(pady=(0, 20))

section_label = ctk.CTkLabel(
    main_frame,
    text="Select Symptoms",
    font=("Arial", 18, "bold")
)
section_label.pack(anchor="w", padx=25, pady=(5, 10))

symptoms_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
symptoms_frame.pack(fill="x", padx=25)

symptom_names = [
    "fever",
    "cough",
    "headache",
    "fatigue",
    "sore_throat",
    "runny_nose",
    "nausea",
    "vomiting",
    "diarrhea",
    "muscle_pain",
    "sneezing",
    "difficulty_breathing"
]

symptom_vars = {}

for index, symptom in enumerate(symptom_names):
    var = ctk.IntVar(value=0)
    symptom_vars[symptom] = var

    checkbox = ctk.CTkCheckBox(
        symptoms_frame,
        text=symptom.replace("_", " ").title(),
        variable=var,
        font=("Arial", 13),
        checkbox_width=20,
        checkbox_height=20
    )

    checkbox.grid(
        row=index // 2,
        column=index % 2,
        sticky="w",
        padx=10,
        pady=9
    )

# Result section
result_label = ctk.CTkLabel(
    main_frame,
    text="Model Prediction: Waiting...",
    font=("Arial", 16, "bold"),
    text_color="#60A5FA"
)
result_label.pack(pady=(25, 15))


def analyze_symptoms():
    symptoms = {
        name: var.get()
        for name, var in symptom_vars.items()
    }

    sample = pd.DataFrame([symptoms])

    prediction = model.predict(sample)[0]

    result_label.configure(
        text=f"Model Prediction: {prediction.replace('_', ' ').title()}"
    )


analyze_button = ctk.CTkButton(
    main_frame,
    text="Analyze Symptoms",
    font=("Arial", 15, "bold"),
    height=45,
    corner_radius=12,
    command=analyze_symptoms
)
analyze_button.pack(fill="x", padx=25, pady=(5, 20))

disclaimer = ctk.CTkLabel(
    main_frame,
    text=(
        "EDUCATIONAL PURPOSE ONLY\n\n"
        "SymptomAI uses a synthetic dataset to demonstrate "
        "Machine Learning classification.\n\n"
        "It does not provide medical diagnoses, medical advice "
        "or reliable health assessments.\n"
        "Predictions are not medically validated and must never "
        "be used for healthcare decisions."
    ),
    font=("Arial", 10),
    text_color="gray",
    wraplength=410,
    justify="center"
)
disclaimer.pack(padx=20, pady=(5, 15))

app.mainloop()
