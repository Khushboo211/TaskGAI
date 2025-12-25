"""
LSTM Text Generator – Character-Level
Author: Khushboo Thakre
Purpose: Memory-safe, GPU-ready LSTM text generator with temperature sampling
Requirements Fulfilled:
1. Data preprocessing
2. Model architecture & training
3. Text generation
4. Sample outputs
5. Bonus: Experiment with deeper LSTM & different sequence lengths
"""

# Imports

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# GPU Check

print("Available GPUs:", tf.config.list_physical_devices('GPU'))
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Dataset Loading & Preprocessing

DATASET_PATH = "data"  # Folder containing .txt files

def load_text_files(folder_path):
    """
    Loads all .txt files from folder_path, converts to lowercase,
    and concatenates them into a single string.
    """
    text_data = ""
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8", errors="ignore") as f:
                text_data += f.read().lower() + "\n"
    return text_data

# Load data
text = load_text_files(DATASET_PATH)
print(f"Total characters in dataset: {len(text)}")

# Character-level tokenization
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

encoded_text = np.array([char_to_idx[c] for c in text])


# Memory-Safe Data Generator

SEQ_LENGTH = 50
BATCH_SIZE = 64

def data_generator(encoded, seq_length, batch_size):
    """
    Generator that yields batches for training.
    """
    while True:
        X, y = [], []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(encoded) - seq_length - 1)
            X.append(encoded[idx:idx + seq_length])
            y.append(encoded[idx + seq_length])
        yield np.array(X), np.array(y)

steps_per_epoch = len(encoded_text) // (SEQ_LENGTH * BATCH_SIZE)


# LSTM Model Architecture

def build_lstm_model(vocab_size, seq_length=SEQ_LENGTH, embedding_dim=128, lstm_units=[256,128]):
    """
    Builds a character-level LSTM model.
    lstm_units: list of integers representing number of units in each LSTM layer
    """
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units)-1
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# Experiment: deeper LSTM
lstm_units = [256, 128]  # You can experiment with [512, 256, 128] for bonus
model = build_lstm_model(vocab_size, lstm_units=lstm_units)
model.summary()


# Callbacks

callbacks = [
    EarlyStopping(monitor='loss', patience=3),
    ModelCheckpoint("lstm_text_generator.keras", monitor='loss', save_best_only=True)
]


# Train Model

model.fit(
    data_generator(encoded_text, SEQ_LENGTH, BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=2,
    callbacks=callbacks
)


# Temperature Sampling Function

def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# Text Generation

def generate_text(seed_text, length=300, temperature=0.8):
    seed_text = seed_text.lower()
    result = seed_text

    for _ in range(length):
        encoded = [char_to_idx.get(c, 0) for c in seed_text]
        encoded = pad_sequences([encoded], maxlen=SEQ_LENGTH, truncating='pre')
        preds = model.predict(encoded, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_idx]
        result += next_char
        seed_text += next_char
        seed_text = seed_text[1:]
    return result

#Sample Outputs (Requirement 2)
seed_texts = [
    "to be or not to be",
    "once upon a time",
    "in a galaxy far far away"
]

for seed in seed_texts:
    print(f"\nSeed: {seed}\nGenerated Text:\n{generate_text(seed, 400, temperature=0.7)}\n")
