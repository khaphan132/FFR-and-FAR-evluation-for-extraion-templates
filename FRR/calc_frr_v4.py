import itertools
import random
import numpy as np

from utils import (
    calc_hamming_dictance,
    get_binary_vector_of_real_vectors,
    split_1_array_to_2_array,
)


################### calculate the FRR ######################
""" 
For each user:
    # get the enrolled vector of this user
    # get all the test vectors of this user
    For each test vector:
	Total_genuine_trial ++
        # compute: intra_hamming = hamming_distance_between current test vs enrolled vector (this intra_hamming is measured in number of bits, not percent)
        For each key size i
            # get the corresponding error toleration (ET) of this key size
            If intra_hamming > ET: False_rejection_times[i]++
        End for
	End for
End for
FRR = False_rejection_times * 100 /  Total_genuine_trial;
"""


def calcFRRs(
    enroll_ebds: dict, 
    test_ebds: dict, 
    vkey_len_arr: np.array, 
    verr_capacity_arr: np.array,
):
    size_key_len_arr = len(vkey_len_arr)

    # vector, same length as the length of vector key size,
    # each element of this vector is the false rejection corresponding to the key size
    false_rejection_times = [0] * size_key_len_arr
    total_genuine_trial = 0
    intra_sum = 0

    for user in enroll_ebds.keys():
        # print(user)
        total_genuine_trial += 1
        intra_hamming = calc_hamming_dictance(enroll_ebds[user], test_ebds[user])
        intra_sum += intra_hamming

        for index_key_size in range(0, size_key_len_arr):
            if intra_hamming > verr_capacity_arr[index_key_size]:
                false_rejection_times[index_key_size] += 1

    print("mean hamming distance when calc FRR: ", intra_sum / total_genuine_trial)
    return np.divide(false_rejection_times, (total_genuine_trial * 1.0) / 100)
