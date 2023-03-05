#Create a model to predict the boiling point of aspirin (acetylsalicylic acid) using thep.
#Then compare the predicted boiling point with the experimental boiling point.

#The database is from chemlb about small molecule drugs.

#Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import sklearn as sk
import tensorflow as tf
#from tf.keras.layers import Dense, Input
#from tf.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




df = pd.read_table("tsv.tsv")

# Convert object columns to float to avoid data type errors
cols_to_convert = ["AlogP", "Polar Surface Area", "HBA", "HBD", "#RO5 Violations", "#Rotatable Bonds", 
                   "QED Weighted", "CX Acidic pKa", "CX Basic pKa", "CX LogP", "CX LogD", "Aromatic Rings", 
                   "Heavy Atoms", "HBA (Lipinski)", "HBD (Lipinski)", "#RO5 Violations (Lipinski)", 
                   "Molecular Weight (Monoisotopic)", "Np Likeness Score"]
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")

# Drop irrelevant columns
cols_to_drop = ["Max Phase", "Inorganic Flag", "Structure Type", "Molecular Species"]
df = df.drop(columns=cols_to_drop)

# Drop rows with missing values
df = df.dropna()

# Rename the Chemical ID column
df = df.rename(columns={"ChEMBL ID": "Chemical ID"})

# Set the Chemical ID column as index
df = df.set_index("Chemical ID")

# Drop duplicates
df = df.drop_duplicates()
#df = pd.read_table('tsv.tsv', error_bad_lines=False)
#print(df.head())
#Check the data
df.info()



# Select relevant features
features = ["Polar Surface Area", "#Rotatable Bonds", "Molecular Weight (Monoisotopic)"]

# Select target variable
target = "#RO5 Violations (Lipinski)"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model architecture
inputs = tf.keras.layers.Input(shape=(len(features),))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)



#------------------Predict the lipinski rules of aspirin------------------
#Aspirin info also gotten from 
aspirin = {"Polar Surface Area": 63.60, "#Rotatable Bonds": 2, "Molecular Weight (Monoisotopic)": 180.0423}
aspirin_scaled = scaler.transform([list(aspirin.values())])
prediction = model.predict(aspirin_scaled)
if prediction >= 0.5:
    print("Aspirin violates Lipinski's rule of five.")
else:
    print("Aspirin does not violate Lipinski's rule of five.")
