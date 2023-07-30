import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.models import load_model
from sklearn.compose import ColumnTransformer


class HealthParams:
    def __init__(self, value):
        self.value = value
        self.next = None


class HealthVitalsLinkedList:
    def __init__(self):
        self.head = None

    def prepend(self, value):
        new_node = HealthParams(value)
        new_node.next = self.head
        self.head = new_node

    def get_recent_fit_scores(self, n):
        recent_fit_scores = []
        current_node = self.head
        while current_node and n > 0:
            recent_fit_scores.append(current_node.value)
            current_node = current_node.next
            n -= 1
        return recent_fit_scores


def fit_transformers(X_train, categorical_cols):
    # Fit the transformer on the training data
    ct = ColumnTransformer(transformers=[(
        'encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
    X_train = ct.fit_transform(X_train)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    return ct, scaler


def predict_fitness_score(input_data, ct, scaler):
    # Load the pre-trained model
    model = load_model(
        r'C:\Users\subra\Dropbox\My PC (LAPTOP-778OGN6L)\Desktop\techne\techne_ANN.h5')

    # Convert gender to numerical values
    input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})

    # Transform the input data using the fitted transformer and scaler
    input_data = ct.transform(input_data)
    input_data = scaler.transform(input_data)

    # Make predictions using the trained model
    predictions = model.predict(input_data).flatten()
    return predictions


def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_user_input(heart_rate_list, sleep_duration_list, systolic_bp_list, diastolic_bp_list, stress_level_list):
    try:
        hr = get_float_input("Enter Heart Rate values: ")
        if pd.notna(hr):
            heart_rate_list.prepend(hr)

        sleep = get_float_input("Enter Sleep Duration values: ")
        if pd.notna(sleep):
            sleep_duration_list.prepend(sleep)

        sys_bp = get_float_input("Enter Systolic Blood Pressure values: ")
        if pd.notna(sys_bp):
            systolic_bp_list.prepend(sys_bp)

        dias_bp = get_float_input("Enter Diastolic Blood Pressure values: ")
        if pd.notna(dias_bp):
            diastolic_bp_list.prepend(dias_bp)

        stress = get_float_input("Enter Stress Level values: ")
        if pd.notna(stress):
            stress_level_list.prepend(stress)

        gender = int(input("Enter Gender (0 for male, 1 for female): "))
        age = get_float_input("Enter Age: ")
        daily_steps = get_float_input("Enter Daily Steps: ")

        # Return user input as a dictionary
        user_input = {
            'Heart Rate': hr,
            'Stress Level': stress,
            'Systolic': sys_bp,
            'Diastolic': dias_bp,
            'Sleep Duration': sleep,
            'Gender': gender,
            'Age': age,
            'Daily Steps': daily_steps,
        }
        return user_input
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return get_user_input(heart_rate_list, sleep_duration_list, systolic_bp_list, diastolic_bp_list, stress_level_list)


def main():
    # Create HealthVitalsLinkedList instances
    heart_rate_list = HealthVitalsLinkedList()
    sleep_duration_list = HealthVitalsLinkedList()
    systolic_bp_list = HealthVitalsLinkedList()
    diastolic_bp_list = HealthVitalsLinkedList()
    stress_level_list = HealthVitalsLinkedList()
    fitness_score_list = HealthVitalsLinkedList()

    # Load the dataset
    df = pd.read_csv('health_vitals_data.csv')
    categorical_cols = ['Gender']
    X_train = df.drop(columns=['Fitness Score'])

    # Fit the transformer on the training data
    ct, scaler = fit_transformers(X_train, categorical_cols)

    # Get user input and store in respective linked lists
    user_input = get_user_input(heart_rate_list, sleep_duration_list,
                                systolic_bp_list, diastolic_bp_list, stress_level_list)
    if user_input is None:
        return

    # Convert user input to DataFrame for prediction
    input_data = pd.DataFrame([user_input])

    prediction = predict_fitness_score(input_data, ct, scaler)
    if prediction is None:
        print("Error: Unable to make a prediction.")
        return

    print("Predicted Fitness Score:", prediction[0])

    # Display the linked lists for each health vital
    def display_linked_list(linked_list, health_vital_name):
        current = linked_list.head
        while current:
            print(f"{health_vital_name}: {current.value}")
            current = current.next
        print()

    display_linked_list(heart_rate_list, "Heart Rate")
    display_linked_list(sleep_duration_list, "Sleep Duration")
    display_linked_list(systolic_bp_list, "Systolic Blood Pressure")
    display_linked_list(diastolic_bp_list, "Diastolic Blood Pressure")
    display_linked_list(stress_level_list, "Stress Level")
    display_linked_list(fitness_score_list, "Fitness Score")


if __name__ == "__main__":
    main()
