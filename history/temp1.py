import tensorflow as tf
import pandas as pd

# Step 1: Load the trained ANN model
model = tf.keras.models.load_model('techne.h5')

# # Step 2: Function to get user input
# def get_user_input():
#     print("Please provide the following data:")
#     heart_rate = float(input("Heart Rate (bpm): "))
#     sleep_duration = float(input("Sleep Duration (hours): "))
#     systolic_bp = float(input("Systolic Blood Pressure (mmHg): "))
#     diastolic_bp = float(input("Diastolic Blood Pressure (mmHg): "))
#     stress_level = float(input("Stress Level (0-10): "))
#     age = int(input("Age: "))
#     gender = input("Gender (Male/Female): ")
#     daily_steps = int(input("Daily Steps: "))
#     return heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps
def get_user_input():
    print("Please provide the following data:")
    heart_rate = float(input("Heart Rate (bpm): "))
    sleep_duration = float(input("Sleep Duration (hours): "))
    systolic_bp = float(input("Systolic Blood Pressure (mmHg): "))
    diastolic_bp = float(input("Diastolic Blood Pressure (mmHg): "))
    stress_level = float(input("Stress Level (0-10): "))
    age = int(input("Age: "))
    gender = int(input("Gender (Male=0/Female=1): "))
    daily_steps = int(input("Daily Steps: "))
    return heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps


# Step 3: Use the trained ANN model for prediction
def predict_fitness_score(heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps):
    # Prepare the input as a NumPy array
    input_data = [[heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps]]

    # Make the prediction using the trained model
    predicted_fitness_score = model.predict(input_data)[0][0]

    return predicted_fitness_score

# Step 4: Display the predicted fitness score to the user
def main():
    heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps = get_user_input()
    fitness_score = predict_fitness_score(heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level,
                                          age, gender, daily_steps)

    print(f"Predicted Fitness Score: {fitness_score}")

if __name__ == "__main__":
    main()
