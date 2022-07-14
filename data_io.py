import array
import numpy as np

from ENUM.path_enum import PATH_ENUM


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
        try:
            ebd = ReadList(embedding_path + "/" + item.replace("wav", "txt"), True)
            embeddings_data[label_dict[item]].append(ebd)
        except:
            print("can not open file: ", item)
        
    return embeddings_data

def write_report_threshold(FRR, FAR, EER, threshold, output_folder):
    f = open(output_folder + "report.txt", "w")
    f.write("Threshold: " + str(threshold))
    f.write("\nFRR: " + str(FRR))
    f.write("\nFAR: " + str(FAR))
    f.write("\nEER: " + str(EER))
    f.close()

def write_report_key_len(FRR, FAR, EER,  key_size, err_capacity, output_folder):
    f = open(output_folder + "report.txt", "w")
    f.write("key_size: " + str(key_size))
    f.write("\nerr_capacity: " + str(err_capacity))
    f.write("\nFRR: " + str(FRR))
    f.write("\nFAR: " + str(FAR))
    f.write("\nEER: " + str(EER))
    f.close()