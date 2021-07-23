#!/usr/bin/env python3

import os
import json
import argparse
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

# import pdb
#
# pdb.set_trace()


def get_files_and_folders(folder_name):
    files = []
    folders = []
    for item in os.scandir(folder_name):
        if item.is_dir():
            folders.append(item.path)
        else:
            files.append(item.path)
    return files, folders


def read_files(files):
    list_lines = []
    for file in files:
        with open(file, "r") as f:
            contents = f.read()
            contents = contents.replace("}", "}\n")
            str_lines = contents.split("\n")
            str_lines = [x for x in str_lines if x != ""]
            list_lines.extend(str_lines)
    return list_lines


def raw_data_from_files(root_folder):
    files, folders = get_files_and_folders(root_folder)
    while len(folders) != 0:
        temp_files = []
        temp_folders = []
        for folder in folders:
            new_files, new_folders = get_files_and_folders(folder)
            temp_files.extend(new_files)
            temp_folders.extend(new_folders)
        files.extend(temp_files)
        folders = temp_folders
    list_lines = read_files(files)
    return list_lines


def group_sessions(list_dicts, key_to_group_by):
    # Author: thefourtheye - https://stackoverflow.com/users/1903116/thefourtheye
    # Source: https://stackoverflow.com/questions/21674331/group-by-multiple-keys-and-summarize-average-values-of-a-list-of-dictionaries/21674471#21674471    grouper = itemgetter("dept", "sku")
    sessions = []
    grouper = itemgetter(key_to_group_by)
    for key, grp in groupby(sorted(list_dicts, key=grouper), grouper):
        temp_dict = dict({"user": key})
        temp_dict["seen_artists"] = []
        temp_dict["likes"] = []
        temp_dict["dislikes"] = []
        for item in grp:
            temp_dict["seen_artists"].append(item["target_id"])
            if item["action"] == "like":
                temp_dict["likes"].append(item["target_id"])
            elif item["action"] == "dislike":
                temp_dict["dislikes"].append(item["target_id"])
            if "filter_type" in item.keys():
                temp_dict["filter_type"] = item["filter_type"]
            else:
                temp_dict["filter_type"] = "old"
        if len(temp_dict["seen_artists"]) > 0 and len(temp_dict["seen_artists"]) == len(
            temp_dict["likes"]
        ) + len(temp_dict["dislikes"]):
            sessions.append(temp_dict)
    return sessions


def filter_sessions(list_lines):
    list_dicts = [json.loads(line) for line in list_lines if line[0] == "{"]
    list_dicts_mm = [
        dict
        for dict in list_dicts
        if "gigstarter" in dict["current_url"]
        and "source_id" in dict.keys()
        and "target_id" in dict.keys()
        and "action" in dict.keys()
    ]

    list_dicts_jive = [
        dict
        for dict in list_dicts
        if "gigstarter" in dict["current_url"]
        and "user_id" in dict.keys()
        and "target_id" in dict.keys()
        and "action" in dict.keys()
    ]

    sessions_mm = group_sessions(list_dicts_mm, "source_id")
    sessions_jive = group_sessions(list_dicts_jive, "user_id")
    sessions = []
    sessions.extend(sessions_mm)
    sessions.extend(sessions_jive)
    return sessions


def lists_filter_types(sessions):
    sessions_old = [
        session
        for session in sessions
        if session["filter_type"] == "old"
        or session["filter_type"] == "band_type"
        or session["filter_type"] == "band_proband"
    ]
    sessions_knn = [
        session
        for session in sessions
        if session["filter_type"] == "band_recommender"
        or "knn" in session["filter_type"]
    ]
    sessions_gru = [session for session in sessions if session["filter_type"] == "gru"]
    return sessions_old, sessions_knn, sessions_gru


def read_embeddings():
    def converter(string):
        """Convert string to np array, used to parse artist features when reading csv file."""
        if string in ("[]", ""):
            return None
        string = string.replace(", ", " ")
        string = string[1:-1]
        values = string.split()
        values = [float(value) for value in values]
        features = np.array(values)
        return features

    musicnn_embeddings = pd.read_csv(
        "embeddings/artists_embeddings.csv",
        converters={"features": converter},
        usecols=["id", "features"],
    )
    timbral_embeddings = pd.read_csv(
        "embeddings/artists_timbral_features11.csv",
        converters={"features": converter},
        usecols=["id", "features"],
    )
    id_to_musicnn = dict(zip(musicnn_embeddings.id, musicnn_embeddings.features))
    id_to_timbral = dict(zip(timbral_embeddings.id, timbral_embeddings.features))
    return id_to_musicnn, id_to_timbral


def get_distance(session, id_to_features):
    embeddings = [
        id_to_features[artist]
        for artist in session["seen_artists"]
        if artist in id_to_features.keys() and id_to_features[artist] is not None
    ]
    if len(embeddings) == 0:
        return float("nan"), float("nan")
    embeddings = np.vstack(embeddings)
    pairwise_euclidean_distances = euclidean_distances(embeddings).flatten()
    pairwise_cosine_distances = cosine_similarity(embeddings).flatten()
    mean_euclidean_distance = np.mean(pairwise_euclidean_distances)
    mean_cosine_distance = np.mean(pairwise_cosine_distances)
    return mean_euclidean_distance, mean_cosine_distance


def get_acc_eucl_cosim(sessions):
    id_to_musicnn, id_to_timbral = read_embeddings()
    accs = []
    eucls_musicnn = []
    cosims_musicnn = []
    eucls_timbral = []
    cosims_timbral = []
    for session in sessions:
        acc = len(session["likes"]) / len(session["seen_artists"])
        eucl_musicnn, cosim_musicnn = get_distance(session, id_to_musicnn)
        eucl_timbral, cosim_timbral = get_distance(session, id_to_timbral)
        accs.append(acc)
        eucls_musicnn.append(eucl_musicnn)
        cosims_musicnn.append(cosim_musicnn)
        eucls_timbral.append(eucl_timbral)
        cosims_timbral.append(cosim_timbral)
    results = {}
    results["mean accuracy"] = np.mean(accs)
    results["mean euclidean distance (MusicNN embeddings)"] = np.nanmean(eucls_musicnn)
    results["mean euclidean distance (timbral embeddings)"] = np.nanmean(eucls_timbral)
    results["mean cosine similarity (MusicNN embeddings)"] = np.nanmean(cosims_musicnn)
    results["mean cosine similarity (timbral embeddings)"] = np.nanmean(cosims_timbral)
    results["std accuracy"] = np.std(accs)
    results["std euclidean distance (MusicNN embeddings)"] = np.nanstd(eucls_musicnn)
    results["std euclidean distance (timbral embeddings)"] = np.nanstd(eucls_timbral)
    results["std cosine similarity (MusicNN embeddings)"] = np.nanstd(cosims_musicnn)
    results["std cosine similarity (timbral embeddings)"] = np.nanstd(cosims_timbral)
    return results


def write_results_to_file(results):
    lines = []
    for algorithm, algorithm_results in results.items():
        lines.append("##########\n{0}\n##########\n\n".format(algorithm))
        for result_name, result_value in algorithm_results.items():
            lines.append("The {0} is {1}.\n".format(result_name, result_value))
        lines.append("\n")
    with open("results.txt", "w+") as file:
        file.writelines(lines)


def main():
    root_folder = args.d
    list_lines = raw_data_from_files(root_folder)
    sessions = filter_sessions(list_lines)
    sessions_old, sessions_knn, sessions_gru = lists_filter_types(sessions)
    results_old = get_acc_eucl_cosim(sessions_old)
    results_knn = get_acc_eucl_cosim(sessions_knn)
    results_gru = get_acc_eucl_cosim(sessions_gru)
    results = {"old": results_old, "knn": results_knn, "gru": results_gru}
    write_results_to_file(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", default="datalake_after", help="directory from which to get data"
    )
    args = parser.parse_args()
    main()
