import os
import numbers
import pickle

import s3fs
import numpy as np
import pandas as pd
from scipy.io import arff


def read_artists_s3():
    """Read latest artist data from S3 and compare to current artist embeddings.

    Returns
    -------
    pandas DataFrame
        Three dataframes denoting changes to be made.

    """
    fs = s3fs.S3FileSystem(profile="jive_uploader")
    with fs.open(
        "s3://gigstarter/jive/recommender/server_data/artists_data.json", "r"
    ) as file:
        df_artists = pd.read_json(file)
        df_artists = df_artists[["id", "country", "youtube_id"]]
    with open("embeddings/ours/old_artists.json", "r") as file:
        df_artists_old = pd.read_json(file)
    merged = df_artists_old.merge(df_artists, indicator=True, how="outer")
    df_to_process = merged[merged["_merge"] == "right_only"][
        ["id", "country", "youtube_id"]
    ]
    df_to_drop = merged[merged["_merge"] == "left_only"][["id"]]
    return df_to_process, df_artists, df_to_drop


def download_mp3s(video_ids):
    """Download audio from videos for artists that need to be processed.

    Parameters
    ----------
    video_ids : list of strings
        Each element is the id of an artist's video.
    """
    if not os.path.exists("embeddings/ours/temp_mp3s"):
        os.makedirs("embeddings/ours/temp_mp3s")
    for video_id in video_ids:
        url = "https://www.youtube.com/watch?v=" + video_id
        path = "embeddings/ours/temp_mp3s/{0}.mp3".format(video_id)
        command = (
            "ffmpeg -y $(youtube-dl --youtube-skip-dash --extract-audio -g '"
            + url
            + "' |sed 's/.*/-ss 00:10 -i &/') -t 00:29 "
            + path
        )
        os.system(command)


def use_marsyas():
    """Use MARSYAS to extract features from audio files that were downloaded.
    Features will be saved in 'mp3s.arff'.
    """
    command1 = "mkcollection -c mp3s.mf -l music ./embeddings/ours/temp_mp3s/"
    command2 = "bextract -sv -fe mp3s.mf -w mp3s.arff"
    command3 = "rm bextract_single.mf"
    command4 = "mv *mp3s.arff mp3s.arff"
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)


def read_marsyas(df_artists):
    """Read contents of the file created with MARSYAS and create dictionaries
    mapping video ids to artist ids and countries.

    Parameters
    ----------
    df_artists : pandas DataFrame
        Dataframe containing information about artists that are being processed.

    Returns
    -------
    tuple, list, dict, dict
        All information needed to create artist embeddings.
    """
    with open("mp3s.arff") as file:
        all_lines = file.readlines()

    video_ids = [line for line in all_lines if line.startswith("% f")]
    video_ids = [(line.split("/")[-1]).split(".")[0] for line in video_ids]

    videoid_to_artistid = dict(zip(df_artists.youtube_id, df_artists.id))
    videoid_to_country = dict(zip(df_artists.youtube_id, df_artists.country))

    data, _ = arff.loadarff("mp3s.arff")

    return data, video_ids, videoid_to_artistid, videoid_to_country


def process_marsyas(data, video_ids, videoid_to_artistid, videoid_to_country):
    """Use data obtained from MARSYAS to create artist embeddings.

    Parameters
    ----------
    data : tuple
        Data read from MARSYAS' file.
    video_ids : type
        List of ids of videos that audio features were extracted from.
    videoid_to_artistid : dict
        Maps video ids to artist ids.
    videoid_to_country : dict
        Maps video ids to country.

    Returns
    -------
    pandas DataFrame
        Dataframe containing created embeddings, with columns ['id', 'features', 'country'],
        first with all features and second with a subset of features.

    """
    features = []
    sub_features = []
    for i, datapoint in enumerate(data):
        video_id = video_ids[i]
        if (
            video_id not in videoid_to_artistid.keys()
            or video_id not in videoid_to_country.keys()
        ):
            continue
        artist_id = videoid_to_artistid[video_id]
        country = videoid_to_country[video_id]
        feature_vector = [
            feature for feature in datapoint if isinstance(feature, numbers.Number)
        ]
        # take a subset of features for GRU model
        feature_subvector = [
            feature_vector[i] for i in [4, 5, 6, 7, 30, 35, 61, 66, 92, 97, 123]
        ]
        features.append([artist_id, feature_vector, country])
        sub_features.append([artist_id, feature_subvector, country])
    df_processed = pd.DataFrame(features)
    df_subprocessed = pd.DataFrame(sub_features)
    return df_processed, df_subprocessed


def delete_marsyas_files():
    """Delete files that were created by MARSYAS and mp3s when they are no longer needed."""
    command1 = "rm mp3s.mf"
    command2 = "rm mp3s.arff"
    command3 = "find embeddings/ours/temp_mp3s -type f -delete"
    os.system(command1)
    os.system(command2)
    os.system(command3)


def process_artists(df_artists):
    """Process each artist in the given dataframe by creating embeddings for them.

    Parameters
    ----------
    df_artists : pandas DataFrame
        Dataframe containing all artists that need to be processed,
        with columns ['id', 'country', 'youtube_id'].

    Returns
    -------
    pandas DataFrame, pandas DataFrame
        Dataframe containing the created embeddings with columns ['id', 'features', 'country'],
        first with all features and second with a subset of features.

    """
    video_ids = [str(artist[2]) for artist in df_artists.values.tolist()]
    download_mp3s(video_ids)
    if os.listdir("embeddings/ours/temp_mp3s") == []:
        return pd.DataFrame(columns=[0, 1, 2])
    use_marsyas()
    data, video_ids, videoid_to_artistid, videoid_to_country = read_marsyas(df_artists)
    df_processed, df_subprocessed = process_marsyas(
        data, video_ids, videoid_to_artistid, videoid_to_country
    )
    delete_marsyas_files()
    return df_processed, df_subprocessed


def update_features_s3_and_local(df_processed, df_subprocessed, id_drop):
    """Update files containing artist embeddings with the new embeddings.

    Parameters
    ----------
    df_processed : pandas DataFrame
        Dataframe containing new embeddings.
    df_subprocessed : pandas DataFrame
        Same as df_processed, but with a subset of features.
    id_drop : list
        List of ids of artists that should be removed from the embeddings.

    Returns
    -------
    pandas DataFrame
        Updated DataFrame with artist 'id', 'features', 'country' info, for subset of features.

    """
    route = (
        "s3://gigstarter/jive/recommender/artist_features/artists_timbral_features.csv"
    )
    local_path = "embeddings/ours/artists_timbral_features.csv"
    local_path_sub = "embeddings/ours/artists_timbral_features11.csv"
    fs = s3fs.S3FileSystem(profile="jive_uploader")
    df_processed.columns = ["id", "features", "country"]

    with fs.open(route, "r") as file:
        df_old = pd.read_csv(file)
        df_old.columns = ["id", "features", "country"]
        df_old = df_old.drop(df_old[df_old["id"].isin(id_drop)].index)
        df_updated = pd.concat([df_old, df_processed])
        df_updated = df_updated.drop_duplicates(subset="id", keep="last")
        df_subupdated = pd.concat([df_old, df_subprocessed])
        df_subupdated = df_subupdated.drop_duplicates(subset="id", keep="last")
    with fs.open(route, "w") as file:
        df_updated.to_csv(file, index=False)
    with open(local_path, "w") as file:
        df_updated.to_csv(file, index=False)
    with open(local_path_sub, "w") as file:
        df_subupdated.to_csv(file, index=False)
    return df_subupdated


def rewrite_old_json(df):
    """Update json that is used to compare current embeddings with artists in database."""
    with open("embeddings/ours/old_artists.json", "w") as file:
        df.to_json(file)


def make_embedding_matrix(df):
    """Save features as numpy array for easier loading. Saves additional file with
    song ids, the order of which corresponds to the embedding_matrix, to keep track
    of what features belong to what song.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with columns 'id', 'features', 'country'.
    """

    df.columns = ["id", "features", "country"]
    song_ids = df["id"].tolist()
    with open("embeddings/ours/artist_ids.data", "wb") as file:
        pickle.dump(song_ids, file)
    data = df["features"].to_numpy()
    embedding_matrix = np.zeros((len(df), 124))
    for idx, features in enumerate(data):
        embedding_matrix[idx, :] = np.array([float(feature) for feature in features])
    np.save("embeddings/ours/embedding_matrix11.npy", embedding_matrix)


def main():
    """Update artist embeddings: timbral features extracted with MARSYAS."""
    df_artists, df_json_old, df_drop = read_artists_s3()
    df_artists = df_artists.dropna()
    id_to_drop = df_drop.id.tolist()
    if len(df_artists) != 0:
        df_processed, df_subprocessed = process_artists(df_artists)
        df_subupdated = update_features_s3_and_local(
            df_processed, df_subprocessed, id_to_drop
        )
        rewrite_old_json(df_json_old)
        make_embedding_matrix(df_subupdated)


if __name__ == "__main__":
    main()

# Use only some features (network was trained on 11 features with std > 0.5)
# 4 5 6 7 30 35 61 66 92 97 123 113


##############################
### Get initial embeddings ###
##############################

# with open('OurStuff/artists.arff') as file:
#     all_lines = file.readlines()
#
# filename_lines = [line for line in all_lines if line.startswith('% f')]
# video_ids = [(line.split('/')[-1]).split('.')[0] for line in filename_lines]
#
# df1 = pd.read_csv('OurStuff/features_musicnn.csv', usecols=['id', 'video_id'])
# df2 = pd.read_csv('OurStuff/artists_embeddings_old.csv', usecols=['id', 'country'])
#
# videoid_to_id = dict(zip(df1.video_id, df1.id))
# id_to_country = dict(zip(df2.id, df2.country))
#
# data, meta = arff.loadarff('OurStuff/artists.arff')
#
# features = []
# old_artists = []
# for i, datapoint in enumerate(data):
#     video_id = video_ids[i]
#     if video_id not in videoid_to_id.keys():
#         continue
#     id = videoid_to_id[video_id]
#     if id not in id_to_country.keys():
#         continue
#     country = id_to_country[id]
#     feature_vector = [feature for feature in datapoint if isinstance(feature, numbers.Number)]
#     features.append([id, feature_vector, country])
#     old_artists.append([id, country, video_id])
#
# df_features = pd.DataFrame(features)
# df_features.to_csv('features.csv', index=False)
#
# df_old_artists = pd.DataFrame(old_artists)
# df_old_artists.to_json('old_artists.json')


#########################
### Save numpy matrix ###
#########################

# def converter(string):
#     """ Convert string to np array, used to parse artist features when reading csv file.
#     """
#     if string in ('[]', ''):
#         return None
#     string = string[1:-1]
#     features = string.split(', ')
#     features = [float(feature) for feature in features]
#     return features
#
# df_artists = pd.read_csv('embeddings/ours/artists_timbral_features.csv')
# ids = df_artists['id'].tolist()
# countries = df_artists['country'].tolist()
# features = df_artists['features'].tolist()
# features = [converter(feature) for feature in features]
#
# used_feature_indices = [4, 5, 6, 7, 30, 35, 61, 66, 92, 97, 123]
# new_features = []
# for feature in features:
#     feature = [feature[i] for i in used_feature_indices]
#     new_features.append(feature)
#
# new_artists = [[ids[i], new_features[i], countries[i]] for i, _ in enumerate(ids)]
# df_new = pd.DataFrame(new_artists)
# df_new.to_csv("artists_timbral_features11.csv", index=False)
