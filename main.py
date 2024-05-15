import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


features_df = pd.read_csv('yolov8_features_1000_images (1).csv')

features_df['Bounding Box'] = features_df['Bounding Box'].apply(eval)  # Convert string to tuple

features_df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(features_df['Bounding Box'].tolist(), index=features_df.index)

features_df = features_df.drop('Bounding Box', axis=1)

X = features_df[['x1', 'y1', 'x2', 'y2', 'Confidence']]
y = features_df['class_labels']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(LSTM(50))  # Additional LSTM layer
model.add(Dense(25, activation='relu'))  # Additional Dense layer
model.add(Dense(1, activation='sigmoid'))  # Modify based on your task

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  # Modify based on your task

history = model.fit(X_train_reshaped, y_train, epochs=1, validation_data=(X_test_reshaped, y_test))

model.save('person.h5')


loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100-2:.2f}%')