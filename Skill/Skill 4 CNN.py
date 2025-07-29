import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam

print("Devices Available:", tf.config.list_physical_devices())

num_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

def run_cnn_experiment(seq_length):
    print(f"\nTraining CNN with sequence length: {seq_length}")
    X_train_pad = pad_sequences(X_train, maxlen=seq_length)
    X_test_pad = pad_sequences(X_test, maxlen=seq_length)

    model = Sequential([
        Embedding(input_dim=num_words, output_dim=32, input_length=seq_length),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train_pad, y_train,
                        epochs=3,
                        batch_size=64,
                        validation_split=0.2,
                        verbose=0)

    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

sequence_lengths = [50, 100, 200, 300, 500]
accuracies = [run_cnn_experiment(seq_len) for seq_len in sequence_lengths]

plt.figure(figsize=(8, 5))
plt.plot(sequence_lengths, accuracies, marker='o', linestyle='-', color="green")
plt.title("Impact of Sequence Length on CNN Accuracy (IMDB)")
plt.xlabel("Sequence Length")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.show()
