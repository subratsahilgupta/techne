#!/usr/bin/env python
# coding: utf-8

# In[208]:


import pandas as pd
df = pd.read_csv('data.csv')


# In[209]:


df


# In[210]:


df.drop(['Person ID', 'Occupation','Quality of Sleep','Physical Activity Level','BMI Category','Sleep Disorder'], axis=1, inplace=True)


# In[211]:


df


# # Fitness Score = (0.3 * Normalized_Heart_Rate) + (0.2 * Normalized_Blood_Pressure) + (0.2 * Normalized_Stress_Level) + (0.3 * Normalized_Sleep_Duration)
# 
# 
# 
# 
# 

# In[212]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Preprocess the 'Blood Pressure' column
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic', 'Diastolic']] = df[['Systolic', 'Diastolic']].astype(float)

# Drop the original 'Blood Pressure' column
df.drop(columns=['Blood Pressure'], inplace=True)

# Normalize the health vitals data (optional)
scaler = MinMaxScaler()
columns_to_normalize = ['Heart Rate', 'Stress Level', 'Systolic', 'Diastolic', 'Sleep Duration']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define the weights
weights = {
    'Heart Rate': 0.3,
    'Stress Level': 0.2,
    'Systolic': 0.2,
    'Diastolic': 0.2,
    'Sleep Duration': 0.3
}

# Calculate the fitness score using the defined weights
df['Fitness Score'] = df[columns_to_normalize].dot(pd.Series(weights))

# Save the updated dataset with the fitness score
df.to_csv('health_vitals_data.csv', index=False)


# In[213]:


df1=pd.read_csv('health_vitals_data.csv')


# In[214]:


df1


# In[215]:


# Perform one-hot encoding for categorical features (if applicable)
df1_encoded = pd.get_dummies(df1, columns=['Gender'], drop_first=True)


# # RandomForestRegressor

# In[218]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Split the data into features (X) and target (y)
X = df1_encoded.drop(columns=['Fitness Score'])
y = df1_encoded['Fitness Score']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Create the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Predict the fitness score on the test set
y_pred = rf_regressor.predict(X_test)


# In[219]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')
print(f'Mean Absolute Error (MAE) {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')


# # assigning weights based on importance(from random forest regressor) of health vitals 

# In[183]:


from sklearn.ensemble import RandomForestRegressor

# Initialize the RandomForestRegressor with desired parameters
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_regressor.feature_importances_

# Pair the feature names with their corresponding importance scores
feature_names = X_train.columns
feature_importance_map = dict(zip(feature_names, feature_importances))

# Calculate feature weights based on the importance scores
total_importance = sum(feature_importances)
feature_weights = {feature: importance / total_importance for feature, importance in feature_importance_map.items()}

# Print the feature weights in descending order
sorted_feature_weights = dict(sorted(feature_weights.items(), key=lambda item: item[1], reverse=True))
for feature, weight in sorted_feature_weights.items():
    print(f"{feature}: {weight}")


# In[189]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming 'Blood Pressure' column exists in the DataFrame
# Preprocess the 'Blood Pressure' column
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic', 'Diastolic']] = df[['Systolic', 'Diastolic']].astype(float)

# Drop the original 'Blood Pressure' column
df.drop(columns=['Blood Pressure'], inplace=True)

# Normalize the health vitals data (optional)
scaler = MinMaxScaler()

#not applied weights for all features

columns_to_normalize = ['Heart Rate', 'Stress Level', 'Systolic', 'Diastolic', 'Sleep Duration']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define the weights
weights = {
    'Systolic': 0.6368006743729114,
    'Stress Level': 0.14602288649718667,
    'Diastolic': 0.11727276152462258,
    'Heart Rate': 0.0495632734836484,
    'Age': 0.024583915420945138,
    'Sleep Duration': 0.012022479653539978,
    'Daily Steps': 0.01088205903789609,
    'Gender_Male': 0.002851950009249738   
}

# Calculate the fitness score using the updated weights
fitness_scores_df = df[columns_to_normalize].multiply(pd.Series(weights), axis=1).sum(axis=1)
df['Fitness Score'] = fitness_scores_df

# Save the updated dataset with the fitness score
df.to_csv('new_health_vitals_data.csv', index=False)


# In[190]:


df2=pd.read_csv('new_health_vitals_data.csv')


# In[191]:


df2


# In[192]:


# Perform one-hot encoding for categorical features (if applicable)
df2_encoded = pd.get_dummies(df2, columns=['Gender'], drop_first=True)


# In[193]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into features (X) and target (y)
X = df1_encoded.drop(columns=['Fitness Score'])
y = df1_encoded['Fitness Score']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Create the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Predict the fitness score on the test set
y_pred = rf_regressor.predict(X_test)


# In[194]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')


# # including weights for all features

# In[164]:


import pandas as pd
df = pd.read_csv('data.csv')


# In[165]:


df


# In[166]:


df.drop(['Person ID', 'Occupation','Quality of Sleep','Physical Activity Level','BMI Category','Sleep Disorder'], axis=1, inplace=True)


# In[167]:


df


# In[168]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming 'Blood Pressure' column exists in the DataFrame
# Preprocess the 'Blood Pressure' column
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic', 'Diastolic']] = df[['Systolic', 'Diastolic']].astype(float)

# Drop the original 'Blood Pressure' column
df.drop(columns=['Blood Pressure'], inplace=True)

# One-hot encode the 'Gender' column
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Normalize the health vitals data (optional)
scaler = MinMaxScaler()
columns_to_normalize = ['Heart Rate', 'Stress Level', 'Systolic', 'Diastolic', 'Sleep Duration', 'Age', 'Daily Steps', 'Gender_Male']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define the weights
weights = {
    'Systolic': 0.6368006743729114,
    'Stress Level': 0.14602288649718667,
    'Diastolic': 0.11727276152462258,
    'Heart Rate': 0.0495632734836484,
    'Age': 0.024583915420945138,
    'Sleep Duration': 0.012022479653539978,
    'Daily Steps': 0.01088205903789609,
    'Gender_Male': 0.002851950009249738
}

# Calculate the fitness score using the updated weights
fitness_scores_df = df[columns_to_normalize].multiply(pd.Series(weights), axis=1).sum(axis=1)
df['Fitness Score'] = fitness_scores_df

# Save the updated dataset with the fitness score
df.to_csv('new_health_vitals_data.csv', index=False)


# In[169]:


df3=pd.read_csv('new_health_vitals_data.csv')


# In[170]:


df3


# In[171]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into features (X) and target (y)
X = df3.drop(columns=['Fitness Score'])
y = df3['Fitness Score']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Create the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Predict the fitness score on the test set
y_pred = rf_regressor.predict(X_test)


# In[172]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')


# # eliminating wight for gender as its wight is 0.003..

# In[159]:


import pandas as pd
df = pd.read_csv('data.csv')

df.drop(['Person ID', 'Occupation','Quality of Sleep','Physical Activity Level','BMI Category','Sleep Disorder'], axis=1, inplace=True)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming 'Blood Pressure' column exists in the DataFrame
# Preprocess the 'Blood Pressure' column
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic', 'Diastolic']] = df[['Systolic', 'Diastolic']].astype(float)

# Drop the original 'Blood Pressure' column
df.drop(columns=['Blood Pressure'], inplace=True)

# One-hot encode the 'Gender' column
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Normalize the health vitals data (optional)
scaler = MinMaxScaler()
columns_to_normalize = ['Heart Rate', 'Stress Level', 'Systolic', 'Diastolic', 'Sleep Duration', 'Age', 'Daily Steps']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define the weights without 'Gender_Male'
weights = {
    'Systolic': 0.6431314800901572,
    'Stress Level': 0.14465438702032984,
    'Diastolic': 0.11199863577174524,
    'Heart Rate': 0.04737099887865282,
    'Age': 0.027366703878293933,
    'Sleep Duration': 0.011601028853718285,
    'Daily Steps': 0.014876765507102895,
}

# Calculate the fitness score using the updated weights
fitness_scores_df = df[columns_to_normalize].multiply(pd.Series(weights), axis=1).sum(axis=1)
df['Fitness Score'] = fitness_scores_df

# Save the updated dataset with the fitness score
df.to_csv('new_health_vitals_data.csv', index=False)


# In[160]:


df4=pd.read_csv('new_health_vitals_data.csv')


# In[161]:


df4


# In[162]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into features (X) and target (y)
X = df4.drop(columns=['Fitness Score'])
y = df4['Fitness Score']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Create the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Predict the fitness score on the test set
y_pred = rf_regressor.predict(X_test)


# In[163]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')


# # Support Vector Regression (SVR)

# In[195]:


df = pd.read_csv('data.csv')

df.drop(['Person ID', 'Occupation','Quality of Sleep','Physical Activity Level','BMI Category','Sleep Disorder'], axis=1, inplace=True)


# In[196]:


df


# In[197]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Preprocess the 'Blood Pressure' column
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic', 'Diastolic']] = df[['Systolic', 'Diastolic']].astype(float)

# Drop the original 'Blood Pressure' column
df.drop(columns=['Blood Pressure'], inplace=True)

# Normalize the health vitals data (optional)
scaler = MinMaxScaler()
columns_to_normalize = ['Heart Rate', 'Stress Level', 'Systolic', 'Diastolic', 'Sleep Duration']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define the weights
weights = {
    'Heart Rate': 0.3,
    'Stress Level': 0.2,
    'Systolic': 0.2,
    'Diastolic': 0.2,
    'Sleep Duration': 0.3
}

# Calculate the fitness score using the defined weights
df['Fitness Score'] = df[columns_to_normalize].dot(pd.Series(weights))

# Save the updated dataset with the fitness score
df.to_csv('new2_health_vitals_data.csv', index=False)


# In[198]:


df5=pd.read_csv('new2_health_vitals_data.csv')


# In[199]:


df5


# In[207]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Preprocess the data and split into features (X) and target variable (y)
# ...

# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=28)

# Create and train the SVR model
svr_regressor = SVR(kernel='linear')  # You can choose different kernels like 'rbf' for non-linear relationships
svr_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared (R2) Score:", r2)


# # GradientBoostingRegressor

# In[243]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv('data.csv')


# In[244]:


df.drop(['Person ID', 'Occupation','Quality of Sleep','Physical Activity Level','BMI Category','Sleep Disorder'], axis=1, inplace=True)


# In[245]:


# Preprocess the 'Blood Pressure' column
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic', 'Diastolic']] = df[['Systolic', 'Diastolic']].astype(float)

# Drop the original 'Blood Pressure' column
df.drop(columns=['Blood Pressure'], inplace=True)

# Normalize the health vitals data (optional)
scaler = MinMaxScaler()
columns_to_normalize = ['Heart Rate', 'Stress Level', 'Systolic', 'Diastolic', 'Sleep Duration']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define the weights
weights = {
    'Heart Rate': 0.3,
    'Stress Level': 0.2,
    'Systolic': 0.2,
    'Diastolic': 0.2,
    'Sleep Duration': 0.3
}

# Calculate the fitness score using the defined weights
df['Fitness Score'] = df[columns_to_normalize].dot(pd.Series(weights))

# Save the updated dataset with the fitness score
df.to_csv('new3_health_vitals_data.csv', index=False)


# In[246]:


df6=pd.read_csv('new3_health_vitals_data.csv')


# In[247]:


df6


# In[249]:


df6 = pd.get_dummies(df6, columns=['Gender'], drop_first=True)


# In[261]:


# Define the features (X) and the target variable (y)
X = df6.drop(columns=['Fitness Score'])
y = df6['Fitness Score']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Initialize the Gradient Boosting Regressor
gbr_regressor = GradientBoostingRegressor(n_estimators=1000, random_state=48)

# Train the model on the training data
gbr_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbr_regressor.predict(X_test)


# In[262]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')


# # deeplearning

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense

# Load the data and preprocess it
# (Assuming you already have df containing the dataset with columns ['Heart Rate', 'Stress Level', 'Systolic', 'Diastolic', 'Sleep Duration', 'Age', 'Daily Steps', 'Gender', 'Fitness Score'])
df = pd.read_csv('data.csv')
# Preprocess the 'Blood Pressure' column
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic', 'Diastolic']] = df[['Systolic', 'Diastolic']].astype(float)

# Drop the original 'Blood Pressure' column and 'Gender' column (as it is not needed for the neural network)
#  df.drop(columns=['Blood Pressure', 'Gender'], inplace=True)
df.drop(['Person ID', 'Occupation','Quality of Sleep','Physical Activity Level','BMI Category','Sleep Disorder','Blood Pressure', 'Gender'], axis=1, inplace=True)

# Normalize the health vitals data (optional)
scaler = MinMaxScaler()
columns_to_normalize = ['Heart Rate', 'Stress Level', 'Systolic', 'Diastolic', 'Sleep Duration', 'Age', 'Daily Steps']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Split the data into features (X) and target (y)
X = df.drop(columns=['Fitness Score']).values
y = df['Fitness Score'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Neural Network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Output layer for regression with linear activation

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




