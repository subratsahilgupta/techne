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
    # Convert to lowercase to handle any case input
    gender = input("Gender (Male/Female): ").lower()
    daily_steps = int(input("Daily Steps: "))
    return heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps

# Step 3: Data preprocessing with feature scaling


def preprocess_input(heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps):
    # Feature scaling to [0, 1] range
    max_hr = 200.0
    min_hr = 40.0
    scaled_heart_rate = (heart_rate - min_hr) / (max_hr - min_hr)

    max_sleep = 12.0
    min_sleep = 4.0
    scaled_sleep_duration = (
        sleep_duration - min_sleep) / (max_sleep - min_sleep)

    max_sys_bp = 180.0
    min_sys_bp = 80.0
    scaled_systolic_bp = (systolic_bp - min_sys_bp) / (max_sys_bp - min_sys_bp)

    max_dia_bp = 100.0
    min_dia_bp = 50.0
    scaled_diastolic_bp = (diastolic_bp - min_dia_bp) / \
        (max_dia_bp - min_dia_bp)

    max_stress = 10.0
    min_stress = 0.0
    scaled_stress_level = (stress_level - min_stress) / \
        (max_stress - min_stress)

    max_age = 100.0
    min_age = 18.0
    scaled_age = (age - min_age) / (max_age - min_age)

    max_daily_steps = 15000.0
    min_daily_steps = 1000.0
    scaled_daily_steps = (daily_steps - min_daily_steps) / \
        (max_daily_steps - min_daily_steps)

    # One-hot encoding for gender (1 for female, 0 for male)
    gender_encoded = 1 if gender == 'female' else 0

    return (
        scaled_heart_rate, scaled_sleep_duration, scaled_systolic_bp, scaled_diastolic_bp,
        scaled_stress_level, scaled_age, gender_encoded, scaled_daily_steps
    )

# Step 4: Use the trained ANN model for prediction


def predict_fitness_score(heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps):
    scaled_hr, scaled_sleep, scaled_sys_bp, scaled_dia_bp, scaled_stress, scaled_age, gender_encoded, scaled_steps = preprocess_input(
        heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps
    )

    # Prepare the input as a NumPy array
    input_data = [[scaled_hr, scaled_sleep, scaled_sys_bp, scaled_dia_bp, scaled_stress,
                   scaled_age, gender_encoded, scaled_steps]]

    # Make the prediction using the trained model
    predicted_fitness_score = model.predict(input_data)[0][0]

    return predicted_fitness_score

# Step 5: Display the predicted fitness score to the user


def main():
    print("Welcome to the Fitness Score Prediction Tool!")
    heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, age, gender, daily_steps = get_user_input()
    fitness_score = predict_fitness_score(heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level,
                                          age, gender, daily_steps)

    print(f"\nPredicted Fitness Score: {fitness_score:.2f}")


if __name__ == "__main__":
    main()
