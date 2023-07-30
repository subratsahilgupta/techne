# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
# from keras.models import load_model
# from sklearn.compose import ColumnTransformer


# class HealthParams:
#     def __init__(self, value):
#         self.value = value
#         self.next = None


# class HealthVitalsLinkedList:
#     def __init__(self):
#         self.head = None

#     def prepend(self, value):
#         new_node = HealthParams(value)
#         new_node.next = self.head
#         self.head = new_node

#     def get_recent_fit_scores(self, n):
#         recent_fit_scores = []
#         current_node = self.head
#         while current_node and n > 0:
#             recent_fit_scores.append(current_node.value)
#             current_node = current_node.next
#             n -= 1
#         return recent_fit_scores


# def get_float_input(prompt):
#     while True:
#         try:
#             value = float(input(prompt))
#             return value
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")


# def predict_fitness_score(input_data):
#     # Load the pre-trained model
#     model = load_model(
#         r'C:\Users\subra\Dropbox\My PC (LAPTOP-778OGN6L)\Desktop\techne\techne_ANN.h5')

#     # Load the dataset
#     df = pd.read_csv('health_vitals_data.csv')

#     # Separate features and target variable
#     X = df.drop(columns=['Fitness Score'])
#     y = df['Fitness Score']

#     # Preprocess categorical columns (like 'Gender') using one-hot encoding
#     categorical_cols = ['Gender']
#     ct = ColumnTransformer(
#         transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
#     X = ct.fit_transform(X)

#     # Normalize the data
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)

#     # Preprocess the input data
#     input_data = pd.DataFrame(input_data, columns=df.drop(
#         columns=['Fitness Score']).columns)

#     if input_data.isna().any().any():
#         print("Warning: NaN values found in the input data. Please ensure all values are provided.")
#         print("Preprocessed Input Data:")
#         print(input_data)
#         return None

#     input_data['Gender'] = input_data['Gender'].astype(
#         int)  # Convert gender back to integer
#     input_data = ct.transform(input_data)
#     input_data = scaler.transform(input_data)

#     # Make predictions using the trained model
#     predictions = model.predict(input_data).flatten()
#     return predictions


# def get_user_input(heart_rate_list, sleep_duration_list, systolic_bp_list, diastolic_bp_list, stress_level_list):
#     try:
#         hr = get_float_input("Enter Heart Rate values: ")
#         if pd.notna(hr):
#             heart_rate_list.prepend(hr)

#         sleep = get_float_input("Enter Sleep Duration values: ")
#         if pd.notna(sleep):
#             sleep_duration_list.prepend(sleep)

#         sys_bp = get_float_input("Enter Systolic Blood Pressure values: ")
#         if pd.notna(sys_bp):
#             systolic_bp_list.prepend(sys_bp)

#         dias_bp = get_float_input("Enter Diastolic Blood Pressure values: ")
#         if pd.notna(dias_bp):
#             diastolic_bp_list.prepend(dias_bp)

#         stress = get_float_input("Enter Stress Level values: ")
#         if pd.notna(stress):
#             stress_level_list.prepend(stress)

#         gender = int(input("Enter Gender (0 for male, 1 for female): "))
#         age = get_float_input("Enter Age: ")
#         daily_steps = get_float_input("Enter Daily Steps: ")

#         # Return user input as a dictionary
#         user_input = {
#             'Heart Rate': hr,
#             'Stress Level': stress,
#             'Systolic': sys_bp,
#             'Diastolic': dias_bp,
#             'Sleep Duration': sleep,
#             'Gender': gender,
#             'Age': age,
#             'Daily Steps': daily_steps,
#         }
#         return user_input
#     except ValueError:
#         print("Invalid input. Please enter a valid number.")
#         return get_user_input(heart_rate_list, sleep_duration_list, systolic_bp_list, diastolic_bp_list, stress_level_list)


# def main():
#     # Create HealthVitalsLinkedList instances
#     heart_rate_list = HealthVitalsLinkedList()
#     sleep_duration_list = HealthVitalsLinkedList()
#     systolic_bp_list = HealthVitalsLinkedList()
#     diastolic_bp_list = HealthVitalsLinkedList()
#     stress_level_list = HealthVitalsLinkedList()
#     fitness_score_list = HealthVitalsLinkedList()

#     # Get user input and store in respective linked lists
#     user_input = get_user_input(heart_rate_list, sleep_duration_list,
#                                 systolic_bp_list, diastolic_bp_list, stress_level_list)
#     if user_input is None:
#         return

#     prediction = predict_fitness_score([user_input])
#     if prediction is None:
#         print("Error: Unable to make a prediction.")
#         return

#     print("Predicted Fitness Score:", prediction[0])

#     # Display the linked lists for each health vital
#     def display_linked_list(linked_list, health_vital_name):
#         current = linked_list.head
#         while current:
#             print(f"{health_vital_name}: {current.value}")
#             current = current.next
#         print()

#     display_linked_list(heart_rate_list, "Heart Rate")
#     display_linked_list(sleep_duration_list, "Sleep Duration")
#     display_linked_list(systolic_bp_list, "Systolic Blood Pressure")
#     display_linked_list(diastolic_bp_list, "Diastolic Blood Pressure")
#     display_linked_list(stress_level_list, "Stress Level")
#     display_linked_list(fitness_score_list, "Fitness Score")


# if __name__ == "__main__":
#     main()


# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
# from keras.models import load_model
# from sklearn.compose import ColumnTransformer


# class HealthParams:
#     def __init__(self, value):
#         self.value = value
#         self.next = None


# class HealthVitalsLinkedList:
#     def __init__(self):
#         self.head = None

#     def prepend(self, value):
#         new_node = HealthParams(value)
#         new_node.next = self.head
#         self.head = new_node

#     def get_recent_fit_scores(self, n):
#         recent_fit_scores = []
#         current_node = self.head
#         while current_node and n > 0:
#             recent_fit_scores.append(current_node.value)
#             current_node = current_node.next
#             n -= 1
#         return recent_fit_scores


# def get_float_input(prompt):
#     while True:
#         try:
#             value = float(input(prompt))
#             return value
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")


# def predict_fitness_score(input_data):
#     # Load the pre-trained model
#     model = load_model(
#         r'C:\Users\subra\Dropbox\My PC (LAPTOP-778OGN6L)\Desktop\techne\techne_ANN.h5')

#     # Load the dataset
#     df = pd.read_csv('health_vitals_data.csv')

#     # Separate features and target variable
#     X = df.drop(columns=['Fitness Score'])
#     y = df['Fitness Score']

#     # Preprocess categorical columns (like 'Gender') using one-hot encoding
#     categorical_cols = ['Gender']
#     ct = ColumnTransformer(
#         transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
#     X = ct.fit_transform(X)

#     # Normalize the data
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)

#     # Preprocess the input data
#     input_data = pd.DataFrame(input_data, columns=df.drop(
#         columns=['Fitness Score']).columns)

#     if input_data.isna().any().any():
#         print("Warning: NaN values found in the input data. Please ensure all values are provided.")
#         print("Preprocessed Input Data:")
#         print(input_data)
#         return None

#     input_data['Gender'] = input_data['Gender'].map(
#         {'Male': 0, 'Female': 1})  # Map gender to numerical values
#     input_data = ct.transform(input_data)
#     input_data = scaler.transform(input_data)

#     # Make predictions using the trained model
#     predictions = model.predict(input_data).flatten()
#     return predictions


# def get_user_input(heart_rate_list, sleep_duration_list, systolic_bp_list, diastolic_bp_list, stress_level_list):
#     try:
#         hr = get_float_input("Enter Heart Rate values: ")
#         if pd.notna(hr):
#             heart_rate_list.prepend(hr)

#         sleep = get_float_input("Enter Sleep Duration values: ")
#         if pd.notna(sleep):
#             sleep_duration_list.prepend(sleep)

#         sys_bp = get_float_input("Enter Systolic Blood Pressure values: ")
#         if pd.notna(sys_bp):
#             systolic_bp_list.prepend(sys_bp)

#         dias_bp = get_float_input("Enter Diastolic Blood Pressure values: ")
#         if pd.notna(dias_bp):
#             diastolic_bp_list.prepend(dias_bp)

#         stress = get_float_input("Enter Stress Level values: ")
#         if pd.notna(stress):
#             stress_level_list.prepend(stress)

#         gender = input("Enter Gender (Male or Female): ").capitalize()
#         age = get_float_input("Enter Age: ")
#         daily_steps = get_float_input("Enter Daily Steps: ")

#         # Return user input as a dictionary
#         user_input = {
#             'Heart Rate': hr,
#             'Stress Level': stress,
#             'Systolic': sys_bp,
#             'Diastolic': dias_bp,
#             'Sleep Duration': sleep,
#             'Gender': gender,
#             'Age': age,
#             'Daily Steps': daily_steps,
#         }
#         return user_input
#     except ValueError:
#         print("Invalid input. Please enter a valid number.")
#         return get_user_input(heart_rate_list, sleep_duration_list, systolic_bp_list, diastolic_bp_list, stress_level_list)


# def main():
#     # Create HealthVitalsLinkedList instances
#     heart_rate_list = HealthVitalsLinkedList()
#     sleep_duration_list = HealthVitalsLinkedList()
#     systolic_bp_list = HealthVitalsLinkedList()
#     diastolic_bp_list = HealthVitalsLinkedList()
#     stress_level_list = HealthVitalsLinkedList()
#     fitness_score_list = HealthVitalsLinkedList()

#     # Get user input and store in respective linked lists
#     user_input = get_user_input(heart_rate_list, sleep_duration_list,
#                                 systolic_bp_list, diastolic_bp_list, stress_level_list)
#     if user_input is None:
#         return

#     prediction = predict_fitness_score([user_input])
#     if prediction is None:
#         print("Error: Unable to make a prediction.")
#         return

#     print("Predicted Fitness Score:", prediction[0])

#     # Display the linked lists for each health vital
#     def display_linked_list(linked_list, health_vital_name):
#         current = linked_list.head
#         while current:
#             print(f"{health_vital_name}: {current.value}")
#             current = current.next
#         print()

#     display_linked_list(heart_rate_list, "Heart Rate")
#     display_linked_list(sleep_duration_list, "Sleep Duration")
#     display_linked_list(systolic_bp_list, "Systolic Blood Pressure")
#     display_linked_list(diastolic_bp_list, "Diastolic Blood Pressure")
#     display_linked_list(stress_level_list, "Stress Level")
#     display_linked_list(fitness_score_list, "Fitness Score")


# if __name__ == "__main__":
#     main()


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


def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def load_trained_model(model_path):
    return load_model(model_path)


def preprocess_input_data(input_data, ct, scaler):
    input_data['Gender'] = input_data['Gender'].map(
        {'Male': 0, 'Female': 1})  # Map gender to numerical values
    input_data = ct.transform(input_data)
    input_data = scaler.transform(input_data)
    return input_data


def predict_fitness_score(model, input_data):  # Pass the model as an argument
    # Load the dataset
    df = pd.read_csv('health_vitals_data.csv')

    # Separate features and target variable
    X = df.drop(columns=['Fitness Score'])
    y = df['Fitness Score']

    # Preprocess categorical columns (like 'Gender') using one-hot encoding
    categorical_cols = ['Gender']
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
    X = ct.fit_transform(X)

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Preprocess the input data
    input_data = pd.DataFrame(input_data, columns=df.drop(
        columns=['Fitness Score']).columns)

    if input_data.isna().any().any():
        print("Warning: NaN values found in the input data. Please ensure all values are provided.")
        print("Preprocessed Input Data:")
        print(input_data)
        return None

    input_data['Gender'] = input_data['Gender'].map(
        {'male': 0, 'female': 1})  # Map gender to numerical values
    input_data = ct.transform(input_data)
    input_data = scaler.transform(input_data)

    # Make predictions using the trained model
    predictions = model.predict(input_data).flatten()
    return predictions




# def predict_fitness_score(model, input_data):
#     # Load the dataset
#     df = pd.read_csv('health_vitals_data.csv')

#     # Separate features and target variable
#     X = df.drop(columns=['Fitness Score'])
#     y = df['Fitness Score']

#     # Preprocess categorical columns (like 'Gender') using one-hot encoding
#     categorical_cols = ['Gender']
#     ct = ColumnTransformer(
#         transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
#     X = ct.fit_transform(X)

#     # Normalize the data
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)

#     input_data = pd.DataFrame(input_data, columns=df.drop(
#         columns=['Fitness Score']).columns)

#     if input_data.isna().any().any():
#         print("Warning: NaN values found in the input data. Please ensure all values are provided.")
#         print("Preprocessed Input Data:")
#         print(input_data)
#         return None

#     input_data = preprocess_input_data(input_data, ct, scaler)

#     # Make predictions using the trained model
#     predictions = model.predict(input_data).flatten()
#     return predictions


# def get_user_input(heart_rate_list, sleep_duration_list, systolic_bp_list, diastolic_bp_list, stress_level_list):
#     try:
#         hr = get_float_input("Enter Heart Rate values: ")
#         if pd.notna(hr):
#             heart_rate_list.prepend(hr)

#         sleep = get_float_input("Enter Sleep Duration values: ")
#         if pd.notna(sleep):
#             sleep_duration_list.prepend(sleep)

#         sys_bp = get_float_input("Enter Systolic Blood Pressure values: ")
#         if pd.notna(sys_bp):
#             systolic_bp_list.prepend(sys_bp)

#         dias_bp = get_float_input("Enter Diastolic Blood Pressure values: ")
#         if pd.notna(dias_bp):
#             diastolic_bp_list.prepend(dias_bp)

#         stress = get_float_input("Enter Stress Level values: ")
#         if pd.notna(stress):
#             stress_level_list.prepend(stress)

#         gender = input("Enter Gender (Male or Female): ").capitalize()
#         while gender not in ['Male', 'Female']:
#             print("Invalid input. Please enter 'Male' or 'Female'.")
#             gender = input("Enter Gender (Male or Female): ").capitalize()

#         age = get_float_input("Enter Age: ")
#         daily_steps = get_float_input("Enter Daily Steps: ")

#         # Return user input as a dictionary
#         user_input = {
#             'Heart Rate': hr,
#             'Stress Level': stress,
#             'Systolic': sys_bp,
#             'Diastolic': dias_bp,
#             'Sleep Duration': sleep,
#             'Gender': gender,
#             'Age': age,
#             'Daily Steps': daily_steps,
#         }
#         return user_input
#     except ValueError:
#         print("Invalid input. Please enter a valid number.")
#         return get_user_input(heart_rate_list, sleep_duration_list, systolic_bp_list, diastolic_bp_list, stress_level_list)

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

        # Convert to lowercase
        gender = input("Enter Gender (Male or Female): ").lower()
        while gender not in ['male', 'female']:
            print("Invalid input. Please enter 'male' or 'female'.")
            # Convert to lowercase
            gender = input("Enter Gender (Male or Female): ").lower()

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

    # Load the pre-trained model
    model_path = r'C:\Users\subra\Dropbox\My PC (LAPTOP-778OGN6L)\Desktop\techne\techne_ANN.h5'
    model = load_trained_model(model_path)

    # Get user input and store in respective linked lists
    user_input = get_user_input(heart_rate_list, sleep_duration_list,
                                systolic_bp_list, diastolic_bp_list, stress_level_list)
    if user_input is None:
        return

    prediction = predict_fitness_score(model, [user_input])
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


