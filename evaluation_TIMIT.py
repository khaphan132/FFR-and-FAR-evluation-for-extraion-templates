import os
import numpy as np
from calc_frr_v3 import calcFRRs
from calc_far_v3 import calcFARs
from data_io import ReadList, getEmbeddingsData
from path_enum import PATH_ENUM
from utils import get_key_len_and_error_capacity_by_size, saveChart2Line


# load embeddings from test folders
# store it as following dictionary
""" 
    embeddings = {
        'user_key_01': [data_embeddings_array],
        'user_key_02': [data_embeddings_array],
        ...
    }
"""

"""  Loading label dictionary """
lab_dict = np.load(PATH_ENUM.TIMIT_CLASS_DICT_FILE.value, allow_pickle=True).item()

"""  test list """
wav_lst_te = ReadList(PATH_ENUM.TIMIT_LIST_TEST.value)
snt_te = len(wav_lst_te)

"""  embedding set that perform evaluating """
path_ebd = PATH_ENUM.TIMIT_EMBEDING_NOTRAIN_V41_HM2.value
embeddings = {}
embeddings = getEmbeddingsData(wav_lst_te, lab_dict, path_ebd)


"""  get vkey_len_arr, verr_capacity_arr """
key_len = 1023
vkey_len_arr, verr_capacity_arr = get_key_len_and_error_capacity_by_size(key_len)  # codeword key_size bits

print("--- test case ---")
print("key_len: ", key_len)
print("embedding set: " + path_ebd.split("/")[-1])


"""  calculate FRR & FAR for extraction templates """
FRR = calcFRRs(embeddings, vkey_len_arr, verr_capacity_arr)
FAR = calcFARs(embeddings, vkey_len_arr, verr_capacity_arr)


"""  save result """
output_folder = "output/output_" + path_ebd.split("/")[-1] + "_" + str(key_len) + "/"
print("output folder: " + output_folder)
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

np.savetxt(output_folder + "frr.txt", FRR, delimiter=",")
np.savetxt(output_folder + "far.txt", FAR, delimiter=",")
saveChart2Line(
    [vkey_len_arr, FRR],
    "FRR",
    [vkey_len_arr, FAR],
    "FAR",
    "key size",
    "error rate (%)",
    path_ebd.split("/")[-1] + " - " + str(key_len) + "-bit codewords - FAR & FRR",
    output_folder + "frr-far.png",
)

print("--- done ---")

