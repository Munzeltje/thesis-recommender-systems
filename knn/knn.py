import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class KNN():
    """ Methods and attributes used by the nearest neighbors approach to make
    recommendations.

    Parameters
    ----------
    artist_sequence : list
        List of artists ids that have already been recommended to the current user.
    like_sequence : list
        List of 0s and 1s denoting which artists in artist_sequence were liked.
    country : string
        Country from which artists will be selected.
    k : int
        The 'k' parameter of k nearest neighbors, the number of similar items
        that will be found. Also the batch size for new recommendations.

    Attributes
    ----------
    shown_artists : list
        List of ids of artists that have been shown to the user.
    liked_artists : list
        List of ids of artists that have been liked by the user.
    prev_sessions : np ndarray of shape (number_of_session, embedding_dim)
        Numpy array of all the embedded previous sessions.
    prev_liked_artists: list
        A list of lists, each nested list containing artists that were liked
        during a previous session. The indices correspond to the indices of prev_sessions.
    artist_ids : list
        A list of ids of all embedded artists that will be considered for recommendations.
    artist_embeddings : np ndarray of shape (number_of_artists, embedding_dim)
        A numpy array containing all the embedded artists. Indices correspond to
        indices of artist_ids.
    current_session_embedding : np array
        The embedding of the current session.
    k

    """
    def __init__(self, artist_sequence, like_sequence, country, k):
        self.k = k
        self.shown_artists = []
        self.liked_artists = []

        sessions_file = 'embeddings/session_embeddings.csv'
        artists_file = 'embeddings/artists_embeddings.csv'

        # Old embeddings
        # def converter(string):
        #     string = re.sub(r"[\n\t\s\[\]]*", "", string)
        #     return np.fromiter(string, dtype=int)

        def converter(string):
            """ Convert string to np array, used to parse artist features when reading csv file.
            """
            if string in ('[]', ''):
                return None
            string = string[1:-1]
            values = string.split()
            values = [float(value) for value in values]
            features = np.array(values)
            return features

        df_sessions = pd.read_csv(sessions_file,
                                  names=['id', 'features', 'liked_artists'],
                                  skiprows=1,
                                  converters={'features':converter, 'liked_artists':eval})
        df_artists = pd.read_csv(artists_file,
                                 names=['id', 'features', 'country'],
                                 skiprows=1,
                                 converters={'features':converter})
        df_artists = df_artists[df_artists['features'].notna()]
        df_artists = df_artists[df_artists['country'] == country]

        self.prev_sessions = np.vstack(df_sessions['features'].to_numpy())
        self.prev_liked_artists = df_sessions['liked_artists'].tolist()

        if len(df_artists) == 0:
            artists_features = np.array([])
        else:
            artists_features = np.vstack(df_artists['features'].to_numpy())
        self.artist_ids = df_artists['id'].tolist()
        self.artist_embeddings = artists_features
        # self.artist_similarities = cosine_similarity(artists_features)

        embedding_dim = self.prev_sessions.shape[1]
        self.current_session_embedding = np.zeros(embedding_dim, dtype=int)
        self.update_embedding_current_session(artist_sequence, like_sequence)

    def update_embedding_current_session(self, artist_sequence, like_sequence):
        """ Update embedding of current session given user's response to last
        artist sequence.

        Parameters
        ----------
        artist_sequence : list of int
            List of artist ids that were shown to user.
        like_sequence : type
            List of likes and dislikes, 0s and 1s, corresponding to artist_sequence.
        """
        for i, rec in enumerate(artist_sequence):
            if rec not in self.artist_ids:
                continue
            if rec not in self.shown_artists:
                self.shown_artists.append(rec)
            response = like_sequence[i]
            artist_embedding = self.artist_embeddings[self.artist_ids.index(rec)].copy()
            embedding_dim = artist_embedding.shape[0]
            if response == 1:
                if rec not in self.liked_artists:
                    self.liked_artists.append(rec)
                number_of_artists = len(self.liked_artists)
                session_embedding_update = self.current_session_embedding[:embedding_dim].copy()
            else:
                number_of_artists = len(self.shown_artists) - len(self.liked_artists)
                session_embedding_update = self.current_session_embedding[embedding_dim:].copy()
            if number_of_artists == 1:
                session_embedding_update = artist_embedding
            else:
                session_embedding_update = np.multiply(session_embedding_update, number_of_artists - 1)
                session_embedding_update = np.add(session_embedding_update, artist_embedding)
                session_embedding_update = np.divide(session_embedding_update, number_of_artists)
            if response == 1:
                updated_embedding = np.hstack([session_embedding_update, self.current_session_embedding[embedding_dim:]])
                self.current_session_embedding = updated_embedding
            else:
                updated_embedding = np.hstack([self.current_session_embedding[:embedding_dim], session_embedding_update])
                self.current_session_embedding = updated_embedding

    def add_random_artists(self, artists, num):
        """ Select random artists that have not been shown before and add them to given list.

        Parameters
        ----------
        artists : list
            List of artists to which random artists will be added.
        num : int
            Number of artists to be selected.

        Returns
        -------
        list
            List with random artists added to it.

        """
        artists = set(artists)
        i = 0
        while len(artists) < num:
            i += 1
            max_index = len(self.artist_ids)
            if max_index == 0:
                break
            artist_index = random.randrange(max_index)
            artist = self.artist_ids[artist_index]
            if artist not in self.shown_artists:
                artists.add(artist)
            if i > 100:
                break
        artists = list(artists)
        return artists

    def get_random_batch(self):
        """ Used for making the first batch of recommendations, which is a list
        of random artists.

        Returns
        -------
        list
            A list of artist ids that will be recommended to the user.

        """
        artists = []
        num = self.k
        artist_recs = self.add_random_artists(artists, num)
        self.shown_artists.extend(artist_recs)
        return artist_recs

    def find_similar_sessions(self):
        """ Finds k sessions that are the most similar to the current session.
        Only sessions with a cosine similarity higher than 0.5 are accepted, if
        not enough can be found will return a shorter or empty list.

        Returns
        -------
        list
            A list of similar sessions.

        """
        current_session = self.current_session_embedding.reshape(1, -1)
        session_similarities = cosine_similarity(self.prev_sessions, current_session)
        similar_sessions = np.argpartition(session_similarities, -self.k, axis=0)[-self.k:]
        similar_sessions = list(similar_sessions.flatten())
        similar_sessions = [session for session in similar_sessions
                            if session_similarities[session] > 0.5]
        return similar_sessions

    def get_artists_similar_session(self):
        """ Finds artists that where liked in sessions similar to the current session.
        Might return a short or empty list if not many artists were found or if found
        artists have already been shown. If too many previously liked artists are
        found, samples k artists.

        Returns
        -------
        list
            List of artists.

        """
        similar_sessions = self.find_similar_sessions()
        prev_liked_artists = set()
        for session in similar_sessions:
            prev_liked_artists.update(self.prev_liked_artists[session])
        prev_liked_artists = list(prev_liked_artists)
        prev_liked_artists = [artist for artist in prev_liked_artists
                              if artist not in self.shown_artists
                              and artist in self.artist_ids]    # remove artists that are no longer in the database or that are from the wrong country
        if len(prev_liked_artists) > self.k:
            prev_liked_artists = random.sample(prev_liked_artists, self.k)
        return prev_liked_artists

    def get_similar_artists(self, artists):
        """ Given a list of artists, find k artists similar to the ones that were given.
        Might return a short or empty list if there are no more artists to be shown.
        If too many artists are found, samples k artists.

        Parameters
        ----------
        artists : list
            List of artists the rest of the artists will be compared to.

        Returns
        -------
        list
            List of similar artists.

        """
        num = self.k + 1        # one similar artist will be the artist itself
        similar_artists = set()
        if len(artists) > self.k:
            artists = random.sample(artists, self.k)
        artists = np.array(artists)
        for artist in artists:
            artist_index = self.artist_ids.index(artist)
            artist_embedding = self.artist_embeddings[artist_index].reshape(1, -1)
            artist_similarities = cosine_similarity(artist_embedding, self.artist_embeddings)
            similar = np.argpartition(artist_similarities[0], -num)[-num:]
            similar = similar.tolist()
            similar = [self.artist_ids[artist_index] for artist_index in list(similar)]
            similar_artists.update(similar)
        similar_artists = list(similar_artists)
        similar_artists = [artist for artist in similar_artists
                           if artist not in self.shown_artists
                           and artist not in artists]
        if len(similar_artists) > self.k:
            similar_artists = random.sample(similar_artists, self.k)
        return similar_artists

    def get_next_recs(self):
        """ Use the embedding and previously liked artists for the current session
        to find the next batch of k recommendations. This is done by getting the
        artists that were liked in previous similar sessions and finding artists
        similar to those artists and to artists that were liked during the current session.

        The chosen artists will be added to self.shown_artists.

        Returns
        -------
        list
            A list of artists ids.

        """
        number_of_recs = self.k - 1     # 1 artist will be chosen randomly
        liked_artists = self.liked_artists
        similar_sessions_artists = self.get_artists_similar_session()

        if len(liked_artists) > self.k:
            liked_artists = liked_artists[-self.k:]
        liked_artists = set(liked_artists)
        liked_artists.update(similar_sessions_artists)
        similar_artists = self.get_similar_artists(list(liked_artists))
        similar_artists = set(similar_artists)
        similar_artists.update(similar_sessions_artists)

        similar_artists = list(similar_artists)
        if len(similar_artists) > number_of_recs:
            similar_artists = random.sample(similar_artists, number_of_recs)

        recs = self.add_random_artists(similar_artists, self.k)
        return recs

def main(artist_sequence, like_sequence, country, k=5):
    """ Takes input from request if not the first batch and selects k artists to
    recommend. If it is the first batch, select artists randomly, otherwise
    use previous likes and dislikes to find other artists that may be relevant
    with a nearest neighbors approach.

    Parameters
    ----------
    artist_sequence : list
        List of artist ids, these are the artists the user has just responded to.
        If there were no previous artists, initialize a new session.
    like_sequence : list
        List of responses, 1 for like and 0 for dislike. Each response corresponds
        to the artist in artist_sequence with the same index.
    country : string
        The country from which artists will be selected.
    k : int
        The k parameter for k nearest neighbors and the batch size.

    Returns
    -------
    list
        List of artist ids, the new recommendations.

    """
    if artist_sequence is None:
        knn = KNN([], [], country, k)
        recs = knn.get_random_batch()
    else:
        knn = KNN(artist_sequence, like_sequence, country, k)
        recs = knn.get_next_recs()
    return recs
