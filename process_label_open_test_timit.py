
import numpy as np
from ENUM.path_enum import PATH_ENUM
from data_io import ReadList


"""  test list """
wav_lst_te = ReadList(PATH_ENUM.TIMIT_LIST_TEST_OPEN.value)

key_dict = {}
lab_open_test = {}
next_id = 0
for item in wav_lst_te:
    temp = item.split('/')
    lab_open_test[item] = temp[2]

np.save("embeddings/timit/timit_list/TIMIT_labels_open_test.npy", lab_open_test)
print('Done!')