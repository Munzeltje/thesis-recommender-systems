import json

import s3fs
import boto3
import numpy as np
import pandas as pd

def get_all_lines_s3_bucket():
    """ Connect to the bucket on Amazon containing the data on previous sessions
    and extract all individual strings of json format.

    Returns
    -------
    list
        A list of unprocessed lines obtained from the bucket.

    """
    amazon_s3 = boto3.resource('s3')
    bucket = 'gs-datalake-bucket'

    all_lines = []

    keys = []

    kwargs = {'Bucket': bucket}
    while True:
        resp = boto3.client('s3').list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            keys.append(obj['Key'])

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

    for k in keys:
        file_contents = amazon_s3.Object(bucket, k).get()['Body'].read().decode("utf-8")
        file_contents = file_contents.replace('}', '}\n')
        lines = file_contents.split('\n')
        all_lines.extend(lines)

    return all_lines

def check_if_interaction(line):
    """ Takes a line obtained from the AWS bucket and checks if it is a valid
    user interaction from musicmatch.

    Parameters
    ----------
    line : string
        Raw line from AWS bucket.

    Returns
    -------
    dict or None
        Either a dictionary of the interaction or None.

    """
    if line == "" or line[0] != '{':
        return None

    line_dict = json.loads(line)
    if 'source_id' not in line_dict.keys() or 'target_id' not in line_dict.keys() or 'action' not in line_dict.keys():
        return None

    return line_dict

def get_artist_id_to_features():
    """ Read artists' embeddings from csv file and creates a dictionary that maps
    each artist's id to that artist's feature vector. Artists without valid features
    are dropped.

    Returns
    -------
    dict
        Mapping artist id to artist embedding.
    """
    def converter(string):
        """ Convert string to np array, used to parse artist features when reading csv file.
        """
        if string == '[]' or string == '':
            return None
        string = string[1:-1]
        values = string.split()
        values = [float(value) for value in values]
        features = np.array(values)
        return features

    df_artists = pd.read_csv('/home/mvp/knn/embeddings/artists_embeddings.csv', converters={'features' : converter})
    df_artists = df_artists[df_artists['features'].notna()]
    artists_id_features = dict(zip(df_artists.id, df_artists.features))
    return artists_id_features

def group_sessions(all_lines, artists_id_features):
    """ Given all lines from S3 bucket, group together all interactions per session
    to obtain lists of liked and disliked artists.

    Parameters
    ----------
    all_lines : list
        A list of raw string obtained from S3 bucket, the interactions by users.
    artists_id_features : dict
        Maps an artist's id to corresponding artist features. Used to check if
        artist in interaction is still valid.

    Returns
    -------
    dict
        Maps session id -> (list of liked artists, list of disliked artists).
    """
    session_id_likes_dislikes = {}
    for line in all_lines:
        interaction = check_if_interaction(line)
        if interaction is not None and int(interaction['target_id']) in artists_id_features.keys():
            if str(interaction['source_id']) in session_id_likes_dislikes.keys():
                session_likes, session_dislikes = session_id_likes_dislikes[str(interaction['source_id'])]
            else:
                session_likes = []
                session_dislikes = []
            if interaction['action'] == 'like':
                session_likes.append(int(interaction['target_id']))
            elif interaction['action'] == 'dislike':
                session_dislikes.append(int(interaction['target_id']))
            session_id_likes_dislikes[str(interaction['source_id'])] = (session_likes, session_dislikes)
    return session_id_likes_dislikes

def embed_sessions(all_lines):
    """ Takes interactions as obtained from S3 bucket and uses them to create a
    a dataframe with each session's id, embeddings and list of liked artists.

    Parameters
    ----------
    all_lines : list
        List of strings, raw interactions from S3.

    Returns
    -------
    pandas dataframe
        Dataframe containing each session's id, embeddings and liked artists.
    """
    artists_id_features = get_artist_id_to_features()
    session_id_likes_dislikes = group_sessions(all_lines, artists_id_features)
    sessions = []

    for session_id in session_id_likes_dislikes.keys():
        session_likes, session_dislikes = session_id_likes_dislikes[session_id]
        session_likes_features = [artists_id_features[artist_id] for artist_id in session_likes
                                  if artist_id in artists_id_features.keys()]
        session_dislikes_features = [artists_id_features[artist_id] for artist_id in session_dislikes
                                  if artist_id in artists_id_features.keys()]
        if len(session_likes_features) > 0:
            session_likes_features = np.vstack(session_likes_features)
            session_likes_mean = np.mean(session_likes_features, axis=0)
        else:
            session_likes_mean = np.zeros(50)
        if len(session_dislikes_features) > 0:
            session_dislikes_features = np.vstack(session_dislikes_features)
            session_dislikes_mean = np.mean(session_dislikes_features, axis=0)
        else:
            session_dislikes_mean = np.zeros(50)

        session_embedding = np.hstack(np.array([session_likes_mean, session_dislikes_mean]))

        sessions.append([session_id, session_embedding, session_likes])
    df_sessions = pd.DataFrame(sessions)
    return df_sessions

def write_features_s3_and_local(df):
    """ Writes obtained features both to local and to S3, DELETING OLD ONES.

    Parameters
    ----------
    df : type
        Dataframe containing the processed sessions.
    """
    fs = s3fs.S3FileSystem(profile='jive_uploader')
    with fs.open('s3://gigstarter/jive/recommender/session_features/session_embeddings.csv', 'w') as f:
        df.to_csv(f, index=False)
    with open('/home/mvp/knn/embeddings/session_embeddings.csv', 'w') as f:
        df.to_csv(f, index=False)

def main():
    boto3.setup_default_session(profile_name='jive_uploader')
    all_lines = get_all_lines_s3_bucket()
    # all_lines = ['{"source_id":"12585","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":18609,"interaction":"click","action":"like","timestamp":1610644611304,"filter_type":"band_type"}', '{"source_id":"12862","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":16499,"interaction":"click","action":"dislike","timestamp":1610644613240,"filter_type":"band_type"}', '{"source_id":"12585","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":5703,"interaction":"click","action":"dislike","timestamp":1610644614406,"filter_type":"band_type"}', '{"source_id":"12585","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":6507,"interaction":"click","action":"like","timestamp":1610644616072,"filter_type":"band_type"}', '{"source_id":"1415","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":15490,"interaction":"click","action":"dislike","timestamp":1610644619980,"filter_type":"band_type"}', '{"source_id":"12862","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":15578,"interaction":"click","action":"like","timestamp":1610644621890,"filter_type":"band_type"}', '{"source_id":"12862","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":12166,"interaction":"click","action":"dislike","timestamp":1610644623490,"filter_type":"band_type"}', '{"source_id":"12862","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":15805,"interaction":"click","action":"like","timestamp":1610644624847,"filter_type":"band_type"}']
    # all_lines = ['{"source_id":"999","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":18609,"interaction":"click","action":"like","timestamp":1610644624847,"filter_type":"band_type"}', '{"source_id":"1415","current_url":"https://www.gigstarter.es/calls/12862/musicmatch?secret=cc0268d7161e069d53230527c4f2d617","target_id":18609,"interaction":"click","action":"like","timestamp":1610644624847,"filter_type":"band_type"}']
    df_processed = embed_sessions(all_lines)
    write_features_s3_and_local(df_processed)
    return "Embeddings updated!"

if __name__ == "__main__":
    main()
