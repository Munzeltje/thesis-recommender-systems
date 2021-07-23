import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def load_artists(country):
    """Load artist ids and embeddings and filter by country.

    Parameters
    ----------
    country : string
        The country from which artists will be selected.

    Returns
    -------
    list, numpy array of shape (number_of_artists, embedding_dim)
        The ids and embeddings of each artist from the given country.
    """
    artist_embeddings = np.load("/home/mvp/knn/embeddings/embedding_matrix11.npy")
    with open("/home/mvp/knn/embeddings/artist_ids.data", "rb") as file:
        artist_ids = pickle.load(file)
    df_artists = pd.read_csv(
        "/home/mvp/knn/embeddings/artists_timbral_features11.csv", usecols=["0", "2"]
    )
    df_artists = df_artists[df_artists["2"] == country]
    indices = df_artists.index.tolist()
    artist_ids = [artist_ids[i] for i in indices]
    artist_embeddings = artist_embeddings[indices]
    return artist_ids, artist_embeddings


def random_batch(artist_sequence, artist_ids):
    """Generate a random batch of artists that have not been shown yet.

    Parameters
    ----------
    artist_sequence : list
        Ids of artists that have already been shown to this user.
    artist_ids : list
        Ids of all artists from the country the user is in.

    Returns
    -------
    list
        Ids of randomly selected artists.
    """
    iteration = 0
    batch = []
    while len(batch) < 5:
        iteration += 1
        if iteration == 100:
            break
        artists_to_sample = 5 - len(batch)
        batch.extend(random.sample(artist_ids, artists_to_sample))
        batch = [artist for artist in batch if artist not in artist_sequence]
    return batch


def create_vectorizer(artist_ids):
    """Create a tf vectorizer object with given artist ids as its vocabulary.
    Adds '' and '[UNK]' tokens automatically.

    Parameters
    ----------
    artist_ids : list
        Ids of all artists in relevant country.

    Returns
    -------
    TextVectorization object
        Maps artist ids to their embedding indices.
    """
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=None, vocabulary=list(map(str, artist_ids))
    )
    return vectorizer


def create_embedding_layer(artist_embeddings):
    """Create embedding layer of GRU with given artist embeddings. Add embeddings
    for '' and '[UNK]' tokens and normalize matrix.

    Parameters
    ----------
    artist_embeddings : np array of shape (number_of_artists, embedding_dim)
        Matrix containing embeddings for each artist.

    Returns
    -------
    keras Embedding layer
        First layer of the GRU that transforms input indices to artist embeddings.
    """
    number_of_artists, embedding_dim = artist_embeddings.shape
    # add embeddings for '', '[UNK]' tokens created by vectorizer
    embedding_matrix = np.zeros((number_of_artists + 2, embedding_dim))
    embedding_matrix[2:] = artist_embeddings
    scaler = StandardScaler()
    embedding_matrix = scaler.fit_transform(embedding_matrix)
    embedding_matrix[0] = np.zeros(embedding_matrix[0].shape)
    embedding_matrix[1] = np.zeros(embedding_matrix[1].shape)
    embedding_layer = tf.keras.layers.Embedding(
        number_of_artists + 2,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )
    return embedding_layer


def create_model(artist_embeddings):
    """Create tensorflow model of Gru4Rec, with an embedding layer containing given
    artist embeddings as its first layer and a loaded trained model following the
    embedding layer.

    Parameters
    ----------
    artist_embeddings : np array of shape (number_of_artists, embedding_dim)
        Matrix containing embeddings for each artist.

    Returns
    -------
    keras Sequential object
        The model with embedding layer containing given artist embeddings followed
        by the trained model.
    """
    embedding_layer = create_embedding_layer(artist_embeddings)
    trained_model = tf.keras.models.load_model(
        "/home/mvp/knn/embeddings/trained_gru", compile=False
    )
    model = tf.keras.Sequential()
    model.add(embedding_layer)
    model.add(trained_model)
    return model


def prepare_model_input(artist_sequence, like_sequence, vectorizer):
    """Create list of artists that were like and feed it to vectorizer to obtain
    input for the model.

    Parameters
    ----------
    artist_sequence : list
        Ids of artists that have been shown to the user.
    like_sequence : type
        Sequence of 1s and 0s denoting likes and dislikes respectively.
    vectorizer : TextVectorization object
        Object used to transform the artist ids to their corresponding embedding indices.

    Returns
    -------
    Tensor
        Vectorized model input, sequence of liked artists.
    """
    liked_artists = [
        [
            [str(artist_sequence[i])]
            for i, rating in enumerate(like_sequence)
            if rating == 1
        ]
    ]
    liked_artists = np.array(liked_artists)
    model_input = vectorizer(liked_artists)
    return model_input


def make_recommendations(model, model_input, artist_ids, artist_embeddings):
    """Predict an artist embedding using the model and find the 5 most similar
    artist embeddings. Then translate those embeddings to artist ids.

    Parameters
    ----------
    model : keras Sequential object
        Gru4Rec model.
    model_input : Tensor
        Vectorized input sequence of liked artists.
    artist_ids : list
        Ids of artists from relevant country, order corresponds to order of embeddings.
    artist_embeddings : np array of shape (number_of_artists, embedding_dim)
        Matrix containing embeddings for each artist.

    Returns
    -------
    list
        Ids of artists that will be recommended next.
    """
    predicted_embedding = model.predict(model_input, batch_size=None)
    most_similar = np.argpartition(
        cosine_similarity(
            predicted_embedding.reshape(1, -1), artist_embeddings
        ).flatten(),
        -5,
    )[-5:]
    recommendations = [artist_ids[embedding_index] for embedding_index in most_similar]
    return recommendations


def main(artist_sequence, like_sequence, country):
    artist_ids, artist_embeddings = load_artists(country)
    if artist_sequence is None or sum(like_sequence) < 5:
        return random_batch(artist_sequence, artist_ids)
    vectorizer = create_vectorizer(artist_ids)
    model = create_model(artist_embeddings)
    model_input = prepare_model_input(artist_sequence, like_sequence, vectorizer)
    recommendations = make_recommendations(
        model, model_input, artist_ids, artist_embeddings
    )
    return recommendations
