import os
import numpy as np
from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dotenv import load_dotenv
load_dotenv()

def build_lstm(input_shape: Tuple[int,int], num_classes:int=2) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save(X: np.ndarray, y: np.ndarray, model_path:str="models/lstm.h5", epochs:int=5, batch_size:int=64):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = build_lstm(input_shape=X.shape[1:])
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=batch_size, verbose=1)
    ypred = model.predict(Xte).argmax(axis=1)
    report = classification_report(yte, ypred)
    print(report)
    model.save(model_path)
    return model_path

def load_model(model_path:str="models/lstm.h5") -> keras.Model:
    return keras.models.load_model(model_path)
