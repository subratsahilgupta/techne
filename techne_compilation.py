import tensorflow as tf

# Step 1: Load the trained ANN model
model = tf.keras.models.load_model('techne.h5')

# Step 2: Function to get user input


def get_user_input():
    print("Please provide the following data:")
    heart_rate = float(input("Heart Rate (bpm): "))
    sleep_duration = float(input("Sleep Duration (hours): "))
    systolic_bp = float(input("Systolic Blood Pressure (mmHg): "))
    diastolic_bp = float(input("Diastolic Blood Pressure (mmHg): "))
    stress_level = float(input("Stress Level (0-10): "))
    age = int(input("Age: "))
    gender = input("Gender (Male/Female): ")
    daily_steps = int(input("Daily Steps: "))
    return heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps

# Step 3: Data preprocessing (include all 8 features, including age and daily steps)


def preprocess_input(heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps):
    # Example: Normalize the data to have zero mean and unit variance
    mean_hr = 75.0
    std_hr = 10.0
    normalized_heart_rate = (heart_rate - mean_hr) / std_hr

    mean_sleep = 7.0
    std_sleep = 1.5
    normalized_sleep_duration = (sleep_duration - mean_sleep) / std_sleep

    mean_sys_bp = 120.0
    std_sys_bp = 10.0
    normalized_systolic_bp = (systolic_bp - mean_sys_bp) / std_sys_bp

    mean_dia_bp = 80.0
    std_dia_bp = 8.0
    normalized_diastolic_bp = (diastolic_bp - mean_dia_bp) / std_dia_bp

    mean_stress = 5.0
    std_stress = 2.0
    normalized_stress_level = (stress_level - mean_stress) / std_stress

    mean_age = 35.0
    std_age = 10.0
    normalized_age = (age - mean_age) / std_age

    mean_daily_steps = 8000.0
    std_daily_steps = 2000.0
    normalized_daily_steps = (daily_steps - mean_daily_steps) / std_daily_steps

    # One-hot encoding for gender (assuming 'Male' or 'Female')
    gender_encoded = 1 if gender.lower() == 'female' else 0

    return (
        normalized_heart_rate, normalized_sleep_duration, normalized_systolic_bp, normalized_diastolic_bp,
        normalized_stress_level, normalized_age, gender_encoded, normalized_daily_steps
    )

# Step 4: Use the trained ANN model for prediction


def predict_fitness_score(heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps):
    normalized_hr, normalized_sleep, normalized_systolic_bp, normalized_dia_bp, normalized_stress, normalized_age, normalized_gender, normalized_steps = preprocess_input(
        heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps
    )

    # Prepare the input as a NumPy array
    input_data = [[normalized_hr, normalized_sleep, normalized_systolic_bp, normalized_dia_bp, normalized_stress,
                   normalized_age, normalized_gender, normalized_steps]]

    # Make the prediction using the trained model
    predicted_fitness_score = model.predict(input_data)[0][0]

    return predicted_fitness_score

# Step 5: Display the predicted fitness score to the user


def main():
    heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps = get_user_input()
    fitness_score = predict_fitness_score(heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level,
                                          age, gender, daily_steps)

    print(f"Predicted Fitness Score: {fitness_score}")


if __name__ == "__main__":
    main()
