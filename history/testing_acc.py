import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load the trained ANN model
model = tf.keras.models.load_model('techne.h5')

# Step 2: Load the test dataset


def load_test_dataset():
    # Load the test dataset from healthvitals_test.csv (or replace with your test data file)
    test_data = pd.read_csv('health_vitals_data.csv')
    return test_data

# Step 3: Data preprocessing for the test dataset


def preprocess_test_data(test_data):
    # Separate features and target column
    X_test = test_data.drop(columns=['Fitness Score'])
    y_test = test_data['Fitness Score']

    # # Normalize/Scale the features using the same mean and standard deviation as used during training
    # mean_hr = 75.0
    # std_hr = 10.0
    # X_test['Heart Rate'] = (
    #     X_test['Heart Rate'] - mean_hr) / std_hr

    # mean_sleep = 7.0
    # std_sleep = 1.5
    # X_test['Sleep Duration'] = (
    #     X_test['Sleep Duration'] - mean_sleep) / std_sleep

    # mean_sys_bp = 120.0
    # std_sys_bp = 10.0
    # X_test['Systolic'] = (
    #     X_test['Systolic'] - mean_sys_bp) / std_sys_bp

    # mean_dia_bp = 80.0
    # std_dia_bp = 8.0
    # X_test['Diastolic'] = (
    #     X_test['Diastolic'] - mean_dia_bp) / std_dia_bp

    # mean_stress = 5.0
    # std_stress = 2.0
    # X_test['Stress Level'] = (X_test['Stress Level'] -
    #                                  mean_stress) / std_stress

    mean_age = 35.0
    std_age = 10.0
    X_test['Age'] = (X_test['Age'] - mean_age) / std_age

    mean_daily_steps = 8000.0
    std_daily_steps = 2000.0
    X_test['Daily Steps'] = (X_test['Daily Steps'] -
                             mean_daily_steps) / std_daily_steps

    # # One-hot encoding for gender (assuming 'Male' or 'Female')
    # X_test['Gender_Male'] = X_test['Gender_Male']

    return X_test, y_test

# Step 4: Use the trained ANN model for prediction on test data


def predict_on_test_data(X_test):
    # Prepare the input as a NumPy array
    input_data = X_test.values

    # Make the prediction using the trained model
    predicted_fitness_scores = model.predict(input_data)

    return predicted_fitness_scores

# Step 5: Calculate and display the accuracy of predictions


def calculate_accuracy(y_test, predicted_fitness_scores):
    # Calculate Mean Squared Error (MSE) to evaluate accuracy
    mse = mean_squared_error(y_test, predicted_fitness_scores)
    accuracy = 1 - mse
    return accuracy


def main():
    test_data = load_test_dataset()
    X_test, y_test = preprocess_test_data(test_data)
    predicted_fitness_scores = predict_on_test_data(X_test)
    accuracy = calculate_accuracy(y_test, predicted_fitness_scores)

    print(f"Accuracy of the predictions: {accuracy:.2f}")


if __name__ == "__main__":
    main()
