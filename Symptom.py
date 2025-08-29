import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def Symptoms(a, b, c, d):
    # List of symptoms and diseases
    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic_patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']
    
     #List of Diseases is listed in list disease.
    disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

    # Read the training data
    df_train = pd.read_csv("Training.csv")
    df_train.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)
    
    # Prepare X_train and y_train
    X_train = df_train[l1]
    y_train = df_train['prognosis']

    # Initialize and train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Prepare input data
    input_data = np.zeros(len(l1))
    for symptom in [a, b, c, d]:
        input_data[l1.index(symptom)] = 1

    # Predict disease
    predicted_index = clf.predict([input_data])[0]
    predicted_disease = disease[predicted_index]

    # Store prediction in database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS SymptomPrediction (Symptom1 TEXT, Symptom2 TEXT, Symptom3 TEXT, Symptom4 TEXT, PredictedDisease TEXT)")
    cursor.execute("INSERT INTO SymptomPrediction (Symptom1, Symptom2, Symptom3, Symptom4, PredictedDisease) VALUES (?, ?, ?, ?, ?)", (a, b, c, d, predicted_disease))
    conn.commit()
    conn.close()

    # Read the testing data
    df_test = pd.read_csv("Testing.csv")
    df_test.replace({'prognosis': {disease[i]: i for i in range(len(disease))}}, inplace=True)

    # Prepare X_test and y_test
    X_test = df_test[l1]
    y_test = df_test['prognosis']

    # Make predictions using the model
    y_pred = clf.predict(X_test)

    # Check accuracy of predictions
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Model Accuracy:", accuracy)

    return predicted_disease
