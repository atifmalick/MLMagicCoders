# -*- coding: utf-8 -*-
"""ECG.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zTpZGyVFgGzQvvvRDIvqm3PE9BcIOf9P
"""

from google.colab import files

# Upload the kaggle.json file (if you haven't already through sidebar)
uploaded = files.upload()

# Move and set permissions
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d shayanfazeli/heartbeat
!unzip heartbeat.zip -d heartbeat_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/content/heartbeat_data/mitbih_train.csv', header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels (0-4)
le = LabelEncoder()
y = le.fit_transform(y)

# Reshape for CNN-LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv1D(64, 5, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

from google.colab import drive
drive.mount('/content/drive')

model.save('/content/drive/MyDrive/ecg_model.h5')

