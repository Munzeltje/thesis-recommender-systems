import os

import s3fs
import librosa
import pandas as pd
import numpy as np
from musicnn.extractor import extractor

# def read_artists_init():
#     """ Used only when obtained the initial embeddings.
#     """
#     with open('older_artists_data.json', 'r') as f:
#         df_artists_old = pd.read_json(f)
#
#     return df_artists_old, df_artists_old, pd.DataFrame(columns=df_artists_old.columns)
#
# def get_features(artists):
#     """ Used only when obtained the initial embeddings.
#     """
#     def converter(string):
#         """ Convert string to np array, used to parse artist features when reading csv file.
#         """
#         if string == '[]' or string == '':
#             return None
#         string = string[1:-1]
#         values = string.split()
#         values = [float(value) for value in values]
#         features = np.array(values)
#         return features
#
#     processed_artists = []
#     # get id and country from artists and use id to get features -> id, features country
#     df_musicnn_features = pd.read_csv('musicnn_features.csv', converters={'features' : converter})
#     artists_id_features = dict(zip(df_musicnn_features.id, df_musicnn_features.features))
#     for artist in artists:
#         artist_id = artist[0]
#         artist_country = artist[2]
#         if artist_id not in artists_id_features.keys():
#             continue
#         artist_features = artists_id_features[artist_id]
#         processed_artists.append([artist_id, artist_features, artist_country])
#     df_processed = pd.DataFrame(processed_artists)
#     return df_processed

def read_artists_s3():
    fs = s3fs.S3FileSystem(profile='jive_uploader')
    with fs.open('s3://gigstarter/jive/recommender/server_data/artists_data.json', 'r') as f:
        df_artists = pd.read_json(f)

    with open('/home/mvp/knn/embeddings/older_artists_data.json', 'r') as f:
        df_artists_old = pd.read_json(f)

    merged = df_artists_old.merge(df_artists, indicator=True, how='outer')
    df_to_process = merged[merged['_merge'] == 'right_only'][['id', 'cached_genres', 'act_int', 'cover_int', 'country', 'youtube_id']]
    df_to_drop = merged[merged['_merge'] == 'left_only'][['id']]

    return df_to_process, df_artists, df_to_drop

def download_mp3(id_video):
    url = "https://www.youtube.com/watch?v=" + str(id_video)
    name = str(id_video) + '.aac'
    command = 'ffmpeg $(youtube-dl --youtube-skip-dash --extract-audio -g \'' + url + '\' |sed \'s/.*/-ss 00:10 -i &/\') -t 00:29 ' + name
    os.system(command)
    return name

def delete_file(f):
    if os.path.exists(f):
        os.remove(f)
    else:
        print("The file does not exist")

def get_feature_genres_video(id_video):
    file_name = download_mp3(id_video)
    if os.path.exists(file_name):
        try:
            y, sr = librosa.core.load(file_name)
            seconds = librosa.get_duration(y=y, sr=sr)
        except:
            delete_file(file_name)
            return []
        if seconds > 15:
            taggram, tags = extractor(file_name, model='MSD_musicnn', extract_features=False)
            tags_likelihood_mean = np.mean(taggram, axis=0)

        else:
            tags_likelihood_mean = []
        delete_file(file_name)
        return tags_likelihood_mean
    else:
        return []

def embed_artists(artists):
    for i, artist in enumerate(artists):
        artists[i].append(get_feature_genres_video(artist[1]))

    df_artists = pd.DataFrame(artists)
    df_artists = df_artists[[0,3,2]]
    return df_artists

# Combine new features with old ones, updating preexisting and concatenating new ones
def update_features_s3_and_local(df_processed, id_drop):
    route = 's3://gigstarter/jive/recommender/artist_features/artists_embeddings.csv'
    local_path = '/home/mvp/knn/embeddings/artists_embeddings.csv'
    fs = s3fs.S3FileSystem(profile='jive_uploader')
    df_processed.columns = ['id', 'features', 'country']

    with fs.open(route, 'r') as f:
        df_old = pd.read_csv(f)
        df_old.columns = ['id', 'features', 'country']
        df_old = df_old.drop(df_old[df_old['id'].isin(id_drop)].index)
        df_updated = pd.concat([df_old, df_processed])
        df_updated = df_updated.drop_duplicates(subset='id', keep='last')
    # with fs.open(route, 'w') as f:
    #     df_updated.to_csv(f, index=False)
    with open(local_path, 'w') as f:
        df_updated.to_csv(f, index=False)

def rewrite_old_json(df):
    with open('/home/mvp/knn/embeddings/older_artists_data.json', 'w') as f:
        df.to_json(f)

def main():
    df_artists, df_json_old, df_drop = read_artists_s3()
    # df_artists, df_json_old, df_drop = read_artists_init()
    df_artists = df_artists.dropna()
    artists = df_artists[['id', 'youtube_id', 'country']].values.tolist()
    # artists = [[1234, '58Ygs4bI1Yo', 'nl'], [1235, '58Ygs4bI1Yo', 'nl']]
    # df_drop = pd.DataFrame(columns=['id', 'features', 'country'])
    id_to_drop = df_drop.id.tolist()
    if len(artists) != 0:
        df_processed = embed_artists(artists)
        # df_processed = get_features(artists)
        update_features_s3_and_local(df_processed, id_to_drop)
        rewrite_old_json(df_json_old)
    return "Embeddings updated"

if __name__ == "__main__":
    main()
