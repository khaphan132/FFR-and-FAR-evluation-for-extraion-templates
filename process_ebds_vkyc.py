import numpy as np
from ENUM.path_enum import PATH_ENUM
from data_io import ReadList


"""  Loading label dictionary """
lab_dict = np.load(PATH_ENUM.VKYC_CLASS_DICT_FILE.value, allow_pickle=True).item()

"""  train list """
wav_lst_train = ReadList(PATH_ENUM.VKYC_LIST_TRAIN.value)
snt_train = len(wav_lst_train)

"""  test list """
wav_lst_test = ReadList(PATH_ENUM.VKYC_LIST_TEST.value)
snt_test = len(wav_lst_test)

""" all list """
wav_lst_all = ReadList(PATH_ENUM.VKYC_LIST_ALL.value)
snt_all = len(wav_lst_all)

""" used list: train list & test list """
wav_lst_used = np.concatenate((wav_lst_train, wav_lst_test))
snt_used = len(wav_lst_used)

evaluation_ebds = [element for element in wav_lst_all if element not in wav_lst_used]

""" analyzed number of eval_ebds per user  """
ebds_per_user = [0]*117
for item in evaluation_ebds:
    ebds_per_user[lab_dict[item]]+=1

path_file = PATH_ENUM.VKYC_LIST_FOLDER.value + '/ebds_per_user.scp'
with open(path_file, 'w') as fp:
    for item in ebds_per_user:
        # write each item on a new line
        fp.write("%s\n" % item)


path_file = PATH_ENUM.VKYC_LIST_FOLDER.value + '/vkyc_eval.scp'
with open(path_file, 'w') as fp:
    for item in evaluation_ebds:
        # write each item on a new line
        fp.write("%s\n" % item)

print("--- done ---")