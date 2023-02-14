# ----------------------------------------------------------------
# Hiding tensorflow warnings, you may skip this step
# ----------------------------------------------------------------

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ----------------------------------------------------------------
# Importing all required libraries
# ----------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import io
import nltk
import json
from keras.preprocessing.text import Tokenizer
from keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    GlobalAveragePooling1D,
    Flatten,
    Dropout,
    GRU,
)
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.layers import Conv1D, MaxPool1D


# ----------------------------------------------------------------
# Reading the json file and processing it
# ----------------------------------------------------------------


filename = "../Data/intents.json"

with open(filename) as data:
    dataset = json.load(data)


def process_data(dataset):
    tags = []
    inputs = []
    responses = {}
    for intent in dataset["intents"]:
        responses[intent["intent"]] = intent["responses"]
        for lines in intent["text"]:
            inputs.append(lines)
            tags.append(intent["intent"])
    return [tags, inputs, responses]


[tags, inputs, responses] = process_data(dataset)


# ----------------------------------------------------------------
# Loading the dataset a dataframe
# ----------------------------------------------------------------

df = pd.DataFrame({"inputs": inputs, "tags": tags})

df.head()

df = df.sample(frac=1)


# ----------------------------------------------------------------
# Adding the preprocessing features
# ----------------------------------------------------------------

import string

df["inputs"] = df["inputs"].apply(
    lambda sequence: [
        ltrs.lower() for ltrs in sequence if ltrs not in string.punctuation
    ]
)

df["inputs"] = df["inputs"].apply(lambda wrd: "".join(wrd))


tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(df["inputs"])
train = tokenizer.texts_to_sequences(df["inputs"])
features = pad_sequences(train)
le = LabelEncoder()
labels = le.fit_transform(df["tags"])

len(features[0])


input_shape = features.shape[1]
print(input_shape)


features.shape


vocabulary = len(tokenizer.word_index)
print("number of unique words : ", vocabulary)
output_length = le.classes_.shape[0]
print("output length: ", output_length)


tokenizer.word_index


# ----------------------------------------------------------------
# Create the RNN
# ----------------------------------------------------------------

seq = Sequential()
seq.add(Input(shape=(features.shape[1])))
seq.add(Embedding(vocabulary + 1, 100))
seq.add(
    Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        bias_regularizer=tf.keras.regularizers.L2(0.0001),
        kernel_regularizer=tf.keras.regularizers.L2(0.0001),
        activity_regularizer=tf.keras.regularizers.L2(0.0001),
    )
)
seq.add(Dropout(0.3))
seq.add(LSTM(32, dropout=0.3, return_sequences=True))
seq.add(LSTM(16, dropout=0.3, return_sequences=False))
seq.add(
    Dense(128, activation="relu", activity_regularizer=tf.keras.regularizers.L2(0.0001))
)
seq.add(Dropout(0.6))
seq.add(
    Dense(
        output_length,
        activation="softmax",
        activity_regularizer=tf.keras.regularizers.L2(0.0001),
    )
)

# !wget https://nlp.stanford.edu/data/glove.6B.zip


seq.layers

# !unzip glove.6B.zip


glove = "glove.6B.100d.txt"
embedding = {}
glove_file = open(glove)
for line in glove_file:
    arr = line.split()
    single_word = arr[0]
    w = np.asarray(arr[1:], dtype="float32")
    embedding[single_word] = w
glove_file.close()
print("Found %s w vectors" % len(embedding))


max_words = vocabulary + 1
word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_words, 100)).astype(object)
for w, i in word_index.items():
    embedding_vector = embedding.get(w)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


seq.layers[0].set_weights([embedding_matrix])
seq.layers[0].trainable = False


seq.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

seq.summary()

from keras.callbacks import TensorBoard, EarlyStopping

earlyStopping = EarlyStopping(
    monitor="loss", patience=400, mode="min", restore_best_weights=True
)

history_training = seq.fit(
    features, labels, epochs=2000, batch_size=64, callbacks=[earlyStopping]
)

# ----------------------------------------------------------------
# Plotting the Runtime of the model
# ----------------------------------------------------------------

import matplotlib as mp


def draw_plot(data, type_data):
    mp.style.use("seaborn")
    plt.figure(figsize=(25, 5))
    plt.plot(data, "darkorange", label="Train")
    plt.xlabel("Epoch")
    plt.ylabel(type_data)
    plt.legend()


draw_plot(history_training.history["accuracy"], "training set accuracy")


draw_plot(history_training.history["loss"], "training set loss")


seq.evaluate(features, labels, batch_size=64)
