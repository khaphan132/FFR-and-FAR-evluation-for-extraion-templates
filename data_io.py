import array
import numpy as np

from path_enum import PATH_ENUM


def ReadList(list_file_path: str, isNumberArray: bool = False):
    f = open(list_file_path, "r")
    lines = f.readlines()
    list_sig = []

    if isNumberArray:
        for x in lines:
            list_sig.append(float(x.rstrip()))
    else:
        for x in lines:
            list_sig.append(x.rstrip())

    f.close()
    return list_sig


def getEmbeddingsData(data_list: array, label_dict: dict, embedding_path: str):
    embeddings_data = {}

    for item in data_list:
        if not label_dict[item] in embeddings_data.keys():
            embeddings_data[label_dict[item]] = []
        embeddings_data[label_dict[item]].append(ReadList(embedding_path + "/" + item.replace("wav", "txt"), True))
    return embeddings_data
