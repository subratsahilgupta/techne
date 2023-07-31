from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the model
model = load_model('techne_ANN.h5')

# Load the scaler objects
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the data from POST request
        data = request.form.to_dict()
        
        # Convert the data to a DataFrame
        df = pd.DataFrame([data])
        
        # Assuming df contains the features (Gender, Age, Sleep Duration, Stress Level, Heart Rate, Daily Steps, Systolic, Diastolic)
        numerical_features = ['Gender', 'Age', 'Sleep Duration', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
        
        # Convert the numerical features to float and reshape
        data_scaled = scaler_X.transform(df[numerical_features].astype(float).values.reshape(1, -1))
        
        # Make predictions using the loaded model
        prediction_scaled = model.predict(data_scaled)

        # Convert the predicted value back to the original scale
        prediction = scaler_y.inverse_transform(prediction_scaled)

        return render_template('index.html', prediction=prediction[0][0])
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
