import numpy as np
from ENUM.path_enum import PATH_ENUM
from data_io import ReadList


"""  Loading label dictionary """
lab_dict = np.load(PATH_ENUM.VKYC_CLASS_DICT_FILE.value, allow_pickle=True).item()


"""  train list """
wav_lst_train = ReadList(PATH_ENUM.VKYC_LIST_TEST.value)
snt_train = len(wav_lst_train)

"""  test list """
wav_lst_test = ReadList(PATH_ENUM.VKYC_LIST_TEST.value)
snt_test = len(wav_lst_test)

""" all list """
wav_lst_test = ReadList(PATH_ENUM.VK.value)
snt_test = len(wav_lst_test)