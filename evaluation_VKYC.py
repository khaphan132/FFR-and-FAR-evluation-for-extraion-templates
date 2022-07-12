import os
import numpy as np
from FRR.calc_frr_v5 import calcFRRs
from FAR.calc_far_v6 import calcFARs
from data_io import ReadList, getEmbeddingsData
from ENUM.path_enum import PATH_ENUM
from utils import calc_average_of_arrays, get_key_len_and_error_capacity_by_size, saveChart2Line, split_enroll_and_test_embeddings


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
lab_dict = np.load(PATH_ENUM.VKYC_CLASS_DICT_FILE.value, allow_pickle=True).item()

"""  test list """
wav_lst_te = ReadList(PATH_ENUM.VKYC_LIST_EVAL.value)
snt_te = len(wav_lst_te)

"""  embedding set that perform evaluating """
path_ebd = PATH_ENUM.VKYC_EMBEDING_V30.value
embeddings = {}
embeddings = getEmbeddingsData(wav_lst_te, lab_dict, path_ebd)

""" split embeddings to 2 sets: enroll & test """
number_of_enroll_embeddings_per_user = 5
enroll_ebds, test_ebds = split_enroll_and_test_embeddings(embeddings, number_of_enroll_embeddings_per_user)

"""  get vkey_len_arr, verr_capacity_arr """
key_len = 255
vkey_len_arr, verr_capacity_arr = get_key_len_and_error_capacity_by_size(key_len)  # codeword key_size bits

print("--- test case ---")
print("key_len: ", key_len)
print("embedding set: " + path_ebd.split("/")[-1])


"""  calculate FRR & FAR for extraction templates """
FRR = calcFRRs(enroll_ebds, test_ebds, vkey_len_arr, verr_capacity_arr)
FAR = calcFARs(test_ebds, vkey_len_arr, verr_capacity_arr)

EER = calc_average_of_arrays([FRR, FAR])
eer_min = np.min(EER)
eer_min_index = np.argmin(EER)
print(eer_min, eer_min_index, vkey_len_arr[eer_min_index], verr_capacity_arr[eer_min_index])

"""  save result """
output_folder = "output_VKYC/output_" + path_ebd.split("/")[-1] + "_" + str(key_len) + "/"
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

