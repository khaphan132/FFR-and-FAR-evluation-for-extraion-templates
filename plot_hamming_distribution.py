from itertools import combinations, permutations
import os
import random
import numpy as np
from ENUM.path_enum import PATH_ENUM, get_TIMIT_path_ebds_open
from data_io import ReadList, getEmbeddingsData
from utils import visualize_distribution


""" 
1 1 -1 1 -1 
1 1 1 1 -1

3 - 2 = 1
2 - 3 = -1
1 - 4 = -3
4 - 1 = 3

len = 5
(len - innner_product(x1, x2))/2
"""


def compute_hamming_dist(x1, x2, dvecdims=512):
    # we have to convert them to binary vectors
    process_x1 = np.sign(x1)
    process_x2 = np.sign(x2)
    return (dvecdims - np.inner(process_x1, process_x2)) / (2 * dvecdims)


"""  Loading openset's label dictionary """
lab_dict = np.load(PATH_ENUM.TIMIT_LABEL_DICT_OPEN_TEST_FILE.value, allow_pickle=True).item()

"""  test list - openset """
wav_lst_te = ReadList(PATH_ENUM.TIMIT_LIST_TEST_OPEN.value)

hm_intra_arr = np.array([])
hm_inter_arr = np.array([])

# evaluate intra
def evaluate_intra(compute_hamming_dist, embeddings, hm_intra_arr, list_users):
    for user in list_users:
        user_embeddings = embeddings[user]
        for pair_ebds in combinations(user_embeddings, 2):
        # compute hamming distance of intra
            hm_res = compute_hamming_dist(pair_ebds[0], pair_ebds[1])
            hm_intra_arr = np.concatenate((hm_intra_arr, hm_res), axis=None)
    return hm_intra_arr

# evaluate inter
def evaluate_inter(embeddings, hm_inter_arr, list_users, number_of_users):
    number_of_testcase_per_user = 20
    for index_user_a in range(number_of_users - 1):
        for index_user_b in range(index_user_a + 1, number_of_users):
            ebds_user_a = embeddings[list_users[index_user_a]].copy()
            ebds_user_b = embeddings[list_users[index_user_b]].copy()
            counter = 1
            all_permutations = list(permutations(range(len(ebds_user_a)), 2))
            random.shuffle(all_permutations)
            for pair_index in all_permutations:
                if counter > number_of_testcase_per_user:
                    break
            # compute hamming distance of intra
                hm_res = compute_hamming_dist(
                embeddings[list_users[index_user_a]][pair_index[0]], 
                embeddings[list_users[index_user_b]][pair_index[1]]
            )
                hm_inter_arr = np.concatenate((hm_inter_arr, hm_res), axis=None)
                counter+=1
    return hm_inter_arr


"""  embedding set that perform evaluating """
def plot_histogram(hm_intra_arr, hm_inter_arr, path_ebd):
    output_folder = "output_TIMIT/output_" + path_ebd.split("/")[-1] + "_" + "threshold_open/"
    print("output folder: " + output_folder)
    try:
        os.stat(output_folder)
    except:
        os.mkdir(output_folder)

    np.savetxt(output_folder + 'hm_intra_arr.text', hm_intra_arr, delimiter='\n')
    np.savetxt(output_folder + 'hm_inter_arr.text', hm_inter_arr, delimiter='\n')
    visualize_distribution(
        hm_intra_arr,
        hm_inter_arr,
        "Hamming Distance Evaluation",
        output_folder + "Hamming_Model-f.png",
        1.0,
        0.01,
        1.0,
        0.2,
    )



for path_ebd in get_TIMIT_path_ebds_open():
    embeddings = {}
    embeddings = getEmbeddingsData(wav_lst_te, lab_dict, path_ebd)
    list_users = list(embeddings.keys())
    number_of_users = len(embeddings.keys())
    hm_intra_arr = evaluate_intra(compute_hamming_dist, embeddings, hm_intra_arr, list_users)
    hm_inter_arr = evaluate_inter(embeddings, hm_inter_arr, list_users, number_of_users)
    plot_histogram(hm_intra_arr, hm_inter_arr, path_ebd)
    
    print("Done!")





