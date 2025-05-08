import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

application = Flask(__name__)
app = application

# Import logistic regression model
model = pickle.load(open('Logistic Regression.pkl', 'rb'))

# Disease mapping (based on your LabelEncoder)
disease_mapping = {
    0: "Paralysis (brain hemorrhage)",
    1: "Hypertension",
    2: "Hepatitis B",
    3: "Impetigo",
    4: "Chronic cholestasis",
    5: "Hepatitis C",
    6: "Typhoid",
    7: "Dimorphic hemorrhoids(piles)",
    8: "Vertigo (Benign paroxysmal Positional Vertigo)",
    9: "Cervical spondylosis",
    10: "Tuberculosis",
    11: "Hyperthyroidism",
    12: "Malaria",
    13: "Gastroenteritis",
    14: "Osteoarthritis",
    15: "Heart attack",
    16: "Dengue",
    17: "Pneumonia",
    18: "Urinary tract infection",
    19: "Hypoglycemia",
    20: "Bronchial Asthma",
    21: "Arthritis",
    22: "Hepatitis D",
    23: "Hypothyroidism",
    24: "Acne",
    25: "GERD",
    26: "Peptic ulcer disease",
    27: "Psoriasis",
    28: "Drug Reaction",
    29: "Diabetes",
    30: "Varicose veins",
    31: "Hepatitis A",
    32: "Hepatitis E",
    33: "Migraine",
    34: "Allergy",
    35: "Jaundice",
    36: "AIDS",
    37: "Alcoholic hepatitis"
}

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            symptoms = {
                'fever': int(request.form.get('fever', 0)),
                'headache': int(request.form.get('headache', 0)),
                'nausea': int(request.form.get('nausea', 0)),
                'vomiting': int(request.form.get('vomiting', 0)),
                'fatigue': int(request.form.get('fatigue', 0)),
                'joint_pain': int(request.form.get('joint_pain', 0)),
                'skin_rash': int(request.form.get('skin_rash', 0)),
                'cough': int(request.form.get('cough', 0)),
                'weight_loss': int(request.form.get('weight_loss', 0)),
                'yellow_eyes': int(request.form.get('yellow_eyes', 0))
            }

            # Convert to numpy array in the correct feature order
            features = np.array([[symptoms['fever'], symptoms['headache'], symptoms['nausea'], 
                                symptoms['vomiting'], symptoms['fatigue'], symptoms['joint_pain'],
                                symptoms['skin_rash'], symptoms['cough'], symptoms['weight_loss'],
                                symptoms['yellow_eyes']]])

            # Make prediction
            prediction = model.predict(features)
            predicted_disease_code = prediction[0]
            predicted_disease = disease_mapping.get(predicted_disease_code, "Unknown Disease")

            return render_template('index.html', 
                                prediction_text=f'Predicted Disease: {predicted_disease}',
                                symptoms=symptoms)

        except Exception as e:
            return render_template('index.html', 
                                prediction_text=f'Error in prediction: {str(e)}')

    # For GET requests, just show the form
    return render_template('index.html')

# API endpoint for external applications
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['fever', 'headache', 'nausea', 'vomiting', 'fatigue', 
                         'joint_pain', 'skin_rash', 'cough', 'weight_loss', 'yellow_eyes']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Prepare features
        features = np.array([[data['fever'], data['headache'], data['nausea'], 
                            data['vomiting'], data['fatigue'], data['joint_pain'],
                            data['skin_rash'], data['cough'], data['weight_loss'],
                            data['yellow_eyes']]])

        # Make prediction
        prediction = model.predict(features)
        predicted_disease_code = prediction[0]
        predicted_disease = disease_mapping.get(predicted_disease_code, "Unknown Disease")

        return jsonify({
            'prediction': predicted_disease,
            'disease_code': int(predicted_disease_code),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)