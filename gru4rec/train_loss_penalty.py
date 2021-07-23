import argparse

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import input_output
import gru4rec


def create_model(embedding_matrix):
    """Create GRU model with given matrix as its embedding layer and custom loss."""
    num_tokens, embedding_dim = embedding_matrix.shape
    scaler = StandardScaler()
    scaled_embedding_matrix = scaler.fit_transform(embedding_matrix)
    scaled_embedding_matrix[0] = np.zeros(embedding_matrix[0].shape)
    scaled_embedding_matrix[1] = np.zeros(embedding_matrix[0].shape)
    embedding_layer = tf.keras.layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(scaled_embedding_matrix),
        trainable=False,
    )
    continue_training = args.c
    if continue_training:
        directory = args.d
        model_name = args.m
        trainable_model = tf.keras.models.load_model(
            "{0}/{1}".format(directory, model_name)
        )
    else:
        number_of_nodes = args.n
        trainable_model = gru4rec.Gru4Rec(embedding_dim, number_of_nodes)
    model = tf.keras.Sequential()
    model.add(embedding_layer)
    model.add(trainable_model)

    def loss_with_penalty(y_true, y_pred):
        cos_sim_loss_function = tf.keras.losses.CosineSimilarity()
        cos_sim_loss = cos_sim_loss_function(y_true, y_pred)
        normalize_pred = tf.math.l2_normalize(y_pred, 1)
        cos_sim_pred = tf.matmul(normalize_pred, normalize_pred, transpose_b=True)
        cos_sim_pred = tf.math.reduce_mean(cos_sim_pred)
        penalty = tf.math.multiply(cos_sim_pred, 3)
        loss = tf.math.add(cos_sim_loss, penalty)
        return loss

    optimizer = args.o
    model.compile(loss=loss_with_penalty, optimizer=optimizer)
    return model


def train_batch(model, sessions, embedding_matrix, vectorizer, i):
    """Train model on part of the data."""
    window_size = args.w
    if window_size > 0:
        batch_size = args.b
        x_train, x_val, y_train, y_val = input_output.process_sessions(
            sessions, vectorizer, window_size=window_size, batch_size=batch_size
        )
    else:
        x_train, x_val, y_train, y_val = input_output.process_sessions(
            sessions, vectorizer
        )
    # targets are not put through model, get embeddings manually
    y_train = embedding_matrix[y_train.astype("int")]
    y_val = embedding_matrix[y_val.astype("int")]

    print("start fit {0}".format(i))
    model.fit(
        x_train, y_train, batch_size=128, epochs=3, validation_data=(x_val, y_val)
    )
    print("finished fit {0}".format(i))
    return model


def train(model, all_sessions, embedding_matrix, vectorizer):
    """Train the model."""
    TOTAL_NUM_SESSIONS = len(all_sessions)
    batch_size = args.b
    for i in range(0, TOTAL_NUM_SESSIONS, batch_size):
        j = i + batch_size
        if j >= TOTAL_NUM_SESSIONS:
            j = TOTAL_NUM_SESSIONS - 1
        sessions = all_sessions[i:j]
        model = train_batch(model, sessions, embedding_matrix, vectorizer, i)
    return model


def save_model(model):
    trained_model = model.layers[1]
    continue_training = args.c
    directory = args.d
    model_name = args.m
    if continue_training:
        path = "{0}{1}2".format(directory, model_name)
    else:
        path = "{0}{1}".format(directory, model_name)
    trained_model.save(path)


def main():
    embedding_matrix, vectorizer = input_output.load_embedding_matrix(
        "embeddings/msd/embedding_matrix_timbre_std_05.npy",
        "embeddings/msd/song_ids_timbre.data",
    )
    print("loaded embeddings and made vectorizer")
    model = create_model(embedding_matrix)
    print("created model")
    continue_bool = args.c
    # not everything fit in memory
    if continue_bool:
        all_sessions = input_output.read_sessions()[400000:]
    else:
        all_sessions = input_output.read_sessions()[:400000]
    print("loaded sessions")
    model = train(model, all_sessions, embedding_matrix, vectorizer)
    save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="number of nodes in GRU")
    parser.add_argument("-b", type=int, default=10000, help="batch size")
    parser.add_argument(
        "-d", default="trained_models/", help="dir trained model will be saved in"
    )
    parser.add_argument("-m", default="latest_model", help="model name")
    parser.add_argument("-o", default="adam", help="optimizer used for training")
    parser.add_argument(
        "-c", default=False, action="store_true", help="continue training"
    )
    parser.add_argument(
        "-w", type=int, default=0, help="window size for input sequences"
    )
    args = parser.parse_args()
    main()
