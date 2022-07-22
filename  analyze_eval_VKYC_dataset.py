import csv
import numpy as np
from requests import delete
from ENUM.path_enum import PATH_ENUM
from data_io import ReadList


# load embeddings from test folders
# store it as following dictionary
def prepare_list_wav():
    """  Loading label dictionary """
    lab_dict = np.load(PATH_ENUM.VKYC_CLASS_DICT_FILE.value, allow_pickle=True).item()

    """  test list """
    wav_lst_te = ReadList(PATH_ENUM.VKYC_LIST_EVAL.value)
    return lab_dict, wav_lst_te


lab_dict, wav_lst_te = prepare_list_wav()

samples_per_user_dict = {}

for item in wav_lst_te:
    if not lab_dict[item] in samples_per_user_dict.keys():
        samples_per_user_dict[lab_dict[item]] = 1
    else:
        samples_per_user_dict[lab_dict[item]] += 1

print("Total test users: ", len(samples_per_user_dict.keys()))

list_categories = ["1-5", "6-19", "20-39", "40-69", "70-99", "100-149", ">150"]
list_bounds = [[1, 5], [6, 19], [20, 39], [40, 69], [70, 99], [100, 149], [150, 1000]]
result = {}
for idx in range(len(list_categories)):
    result[list_categories[idx]] = 0
    for item in samples_per_user_dict:
        if samples_per_user_dict[item] >= list_bounds[idx][0] and samples_per_user_dict[item] <= list_bounds[idx][1]:
            result[list_categories[idx]] += 1

print(result)


csv_file = "embeddings/vkyc/vkyc_v2_list/analyzing_eval_ebds.csv"
try:
    with open(csv_file, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result.keys())
        writer.writeheader()
        for data in [result]:
            writer.writerow(data)
except IOError:
    print("I/O error")
