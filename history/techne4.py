# -*- coding: utf-8 -*-
"""techne4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KX0jaf-DX69_Gv417NkIkCmx059nZg6l
"""

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

def get_user_input():
    heart_rate = get_float_input("Enter Heart Rate: ")
    sleep_duration = get_float_input("Enter Sleep Duration: ")
    systolic_bp = get_float_input("Enter Systolic Blood Pressure: ")
    diastolic_bp = get_float_input("Enter Diastolic Blood Pressure: ")
    stress_level = get_float_input("Enter Stress Level: ")
    fitness_score = get_float_input("Enter Fitness Score: ")

    return heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, fitness_score

def main():
    heart_rate_list = HealthVitalsLinkedList()
    sleep_duration_list = HealthVitalsLinkedList()
    systolic_bp_list = HealthVitalsLinkedList()
    diastolic_bp_list = HealthVitalsLinkedList()
    stress_level_list = HealthVitalsLinkedList()
    fitness_score_list = HealthVitalsLinkedList()

    while True:
        heart_rate, sleep_duration, systolic_bp, diastolic_bp, stress_level, fitness_score = get_user_input()

        heart_rate_list.prepend(heart_rate)
        sleep_duration_list.prepend(sleep_duration)
        systolic_bp_list.prepend(systolic_bp)
        diastolic_bp_list.prepend(diastolic_bp)
        stress_level_list.prepend(stress_level)
        fitness_score_list.prepend(fitness_score)

        choice = input("Do you want to enter more health parameters? (y/n): ")
        if choice.lower() != 'y':
            break

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

    n = int(input("Enter the number of recent fitness scores to extract: "))
    recent_fit_scores = fitness_score_list.get_recent_fit_scores(n)
    print(f"Recent {n} fitness scores: {recent_fit_scores}")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
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
    def calculate_rolling_window_avg(self, n):
        fit_scores = self.get_recent_fit_scores(n)
        return sum(fit_scores) / len(fit_scores)

    def calculate_rolling_window_avg(self, n):
        fit_scores = self.get_recent_fit_scores(n)
        window_size = min(n, len(fit_scores))
        return sum(fit_scores[-window_size:]) / window_size

    def get_float_input(prompt):
        while True:
            try:
                value = float(input(prompt))
                return value
            except ValueError:
                print("Invalid input. Please enter a valid number.")

def main():
    heart_rate_list = HealthVitalsLinkedList()
    sleep_duration_list = HealthVitalsLinkedList()
    systolic_bp_list = HealthVitalsLinkedList()
    diastolic_bp_list = HealthVitalsLinkedList()
    stress_level_list = HealthVitalsLinkedList()
    fitness_score_list = HealthVitalsLinkedList()

    while True:
        heart_rate = get_float_input("Enter Heart Rate: ")
        sleep_duration = get_float_input("Enter Sleep Duration: ")
        systolic_bp = get_float_input("Enter Systolic Blood Pressure: ")
        diastolic_bp = get_float_input("Enter Diastolic Blood Pressure: ")
        stress_level = get_float_input("Enter Stress Level: ")
        fitness_score = get_float_input("Enter Fitness Score: ")

        heart_rate_list.prepend(heart_rate)
        sleep_duration_list.prepend(sleep_duration)
        systolic_bp_list.prepend(systolic_bp)
        diastolic_bp_list.prepend(diastolic_bp)
        stress_level_list.prepend(stress_level)
        fitness_score_list.prepend(fitness_score)

        choice = input("Do you want to enter more health parameters? (y/n): ")
        if choice.lower() != 'y':
            break

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

    # Calculate and display the overall fitness score
    n = int(input("Enter the number of fitness scores to consider for rolling window average: "))
    overall_fitness_score = fitness_score_list.calculate_rolling_window_avg(n)
    print(f"Overall Fitness Score: {overall_fitness_score}")

    def plot_recent_fit_scores(recent_fit_scores):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(recent_fit_scores)), recent_fit_scores, marker='o', linestyle='-')
        plt.xlabel('Time')
        plt.ylabel('Fitness Score')
        plt.title('Recent Fitness Scores')
        plt.grid(True)
        plt.show()

    #n = int(input("Enter the number of fitness scores to consider for the plot: "))
    recent_fit_scores = fitness_score_list.get_recent_fit_scores(n)
    plot_recent_fit_scores(recent_fit_scores)


if __name__ == "__main__":
    main()







