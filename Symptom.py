import sqlite3
import pandas as pd
import numpy as np

#importing the libraries for the machine learning models
from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier
from sklearn.svm import SVC #Support Vector Classifier
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes

#importing the library for the accuracy score
from sklearn.metrics import accuracy_score

# Global variables to store trained models (train once, reuse many times)
_trained_models = None
_best_model = None
_best_model_name = None
_best_accuracy = None
_l1 = None
_disease = None

def Symptoms(a, b, c, d):
    global _trained_models, _best_model, _best_model_name, _best_accuracy, _l1, _disease
    
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

    # Store globally for reuse
    _l1 = l1
    _disease = disease

    # Only train models if they haven't been trained yet
    if _trained_models is None:
        print("Training models for the first time...")
        
        # Read the training data
        df_train = pd.read_csv("Training.csv")
        prognosis_mapping = {disease[i]: i for i in range(len(disease))}
        df_train['prognosis'] = df_train['prognosis'].map(prognosis_mapping).astype(int)
        
        # Prepare X_train and y_train
        X_train = df_train[l1]
        y_train = df_train['prognosis']

        # Initialize candidate models
        candidate_models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
            'SVC': SVC(kernel='rbf', probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, n_jobs=None, random_state=42),
            'GaussianNB': GaussianNB()
        }

        # Read the testing data and prepare test sets
        df_test = pd.read_csv("Testing.csv")
        df_test['prognosis'] = df_test['prognosis'].map(prognosis_mapping).astype(int)
        X_test = df_test[l1]
        y_test = df_test['prognosis']

        # Train, evaluate and select best model by accuracy on Testing.csv
        model_name_to_accuracy = {}
        best_model_name = None
        best_model = None
        best_accuracy = -1.0

        for name, model in candidate_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            model_name_to_accuracy[name] = acc
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_model = model

        # Store globally for reuse
        _trained_models = model_name_to_accuracy
        _best_model = best_model
        _best_model_name = best_model_name
        _best_accuracy = best_accuracy
        
        print("--------------------------------")
        print("Model accuracies:")
        for name, acc in model_name_to_accuracy.items():
            print(f"  - {name}: {acc:.4f}")
        print("--------------------------------")
    else:
        print("Using pre-trained models...")
        print(f"Best model: {_best_model_name} (Accuracy: {_best_accuracy:.4f})")

    # Prepare input data
    input_data = np.zeros(len(_l1))
    for symptom in [a, b, c, d]:
        try:
            input_data[_l1.index(symptom)] = 1
        except ValueError:
            print(f"Warning: Symptom '{symptom}' not found in known symptoms list")

    # Predict disease using the best model (with feature names to avoid sklearn warnings)
    input_df = pd.DataFrame([input_data], columns=_l1)
    predicted_index = _best_model.predict(input_df)[0]
    predicted_disease = _disease[predicted_index]

    # Log inputs and prediction
    print("Input symptoms:", "Symptom 1:", a, "Symptom 2:", b, "Symptom 3:", c, "Symptom 4:", d)
    print("Predicted disease:", predicted_disease)

    # Store prediction in database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS SymptomPrediction (Symptom1 TEXT, Symptom2 TEXT, Symptom3 TEXT, Symptom4 TEXT, PredictedDisease TEXT)")
    try:
        cursor.execute("INSERT INTO SymptomPrediction (Symptom1, Symptom2, Symptom3, Symptom4, PredictedDisease) VALUES (?, ?, ?, ?, ?)", (a, b, c, d, predicted_disease))
        conn.commit()
        print("DB insert status: success")
    except Exception as db_err:
        print("DB insert status: failed", db_err)
    finally:
        conn.close()

    # Log best model and its accuracy
    print("Selected model:", _best_model_name, "Accuracy:", _best_accuracy)
    print("--------------------------------")

    return predicted_disease
