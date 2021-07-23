import csv
import pickle
import random

import more_itertools
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_features(path):
    with open(path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        id_to_features = {
            row[0]: np.fromiter(row[1][1:-1].split(","), dtype=float) for row in reader
        }
    return id_to_features

def get_vectorizer(id_to_features):
    vocabulary = ['', '[UNK]']
    vocabulary.extend(list(id_to_features.keys()))

    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=None,
        vocabulary=vocabulary
    )
    return vectorizer

def get_embedding_matrix(path):
    id_to_features = get_features(path)
    vectorizer = get_vectorizer(id_to_features)

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    NUM_TOKENS = len(voc)
    EMBEDDING_DIM = 124
    embedding_matrix = np.zeros((NUM_TOKENS, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = id_to_features.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # scaler = StandardScaler()
    # embedding_matrix = scaler.fit_transform(embedding_matrix)
    # embedding_matrix[0] = np.zeros(embedding_matrix[0].shape)
    # embedding_matrix[1] = np.zeros(embedding_matrix[0].shape)
    return embedding_matrix, vectorizer

def load_embedding_matrix(path_embedding_matrix, path_song_ids):
    song_embeddings = np.load(path_embedding_matrix)
    number_of_songs, embedding_dim = song_embeddings.shape
    embedding_matrix = np.zeros((number_of_songs+2, embedding_dim))
    embedding_matrix[2:] = song_embeddings
    # scaler = StandardScaler()
    # embedding_matrix = scaler.fit_transform(embedding_matrix)
    # embedding_matrix[0] = np.zeros(embedding_matrix[0].shape)
    # embedding_matrix[1] = np.zeros(embedding_matrix[0].shape)
    with open(path_song_ids, 'rb') as file:
        song_ids = pickle.load(file)

    vocabulary = list(map(str, song_ids))
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=None,
        vocabulary=vocabulary
    )
    return embedding_matrix, vectorizer

def read_sessions():
    with open("embeddings/msd/filtered_listening_sessions_train.txt", "r") as file:
        sessions = [session.split() for session in file.readlines()]
    return sessions

def process_sessions(sessions, vectorizer, test=False, window_size=None, batch_size=100000):
    if window_size is not None:
        windows = [list(more_itertools.windowed(session, n=window_size)) for session in sessions]
        windows = [list(y) for x in windows for y in x if None not in list(y)]
        if len(windows) > batch_size:
            random.seed(3)
            windows = random.sample(windows, batch_size)
        data = [" ".join(window[:-1]) for window in windows]
        targets = [window[-1] for window in windows]
    else:
        data = [" ".join(session[:-1]) for session in sessions]
        targets = [session[-1] for session in sessions]
    data = vectorizer(np.array([[string] for string in data])).numpy()
    data = data.astype('float64')
    targets = vectorizer(np.array([[string] for string in targets])).numpy()
    targets = targets.astype('float64')
    idx_to_remove1 = np.where(data > 973589)
    data = np.delete(data, idx_to_remove1, axis=0)
    targets = np.delete(targets, idx_to_remove1, axis=0)
    idx_to_remove2 = np.where(targets > 973589)
    data = np.delete(data, idx_to_remove2, axis=0)
    targets = np.delete(targets, idx_to_remove2, axis=0)
    if test:
        return data, targets
    x_train, x_val, y_train, y_val = train_test_split(
        data, targets, test_size=0.15, random_state=4
    )
    return x_train, x_val, y_train, y_val
