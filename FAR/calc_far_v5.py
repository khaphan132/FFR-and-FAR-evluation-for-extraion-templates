import numpy as np

from utils import (
    calc_hamming_dictance,
    get_binary_vector_of_real_vectors,
)

################### calculate the FAR ######################
""" 
For each user:
    # get the enrolled vector of this user
    # get all the test vectors of other users
    For each test vector:
        Total_imposter_trial ++
        # compute: intra_hamming = hamming_distance_between current test vs enrolled vector (this intra_hamming is measured in number of bits, not percent)
        For each key size i
            # get the corresponding error toleration (ET) of this key size
            If intra_hamming <= ET: False_acceptance_times[i]++
        End for
	End for
End for
FAR = False_ acceptance _times * 100 /  Total_imposter_trial;
"""


def calcFARs(
    test_ebds: dict,
    vkey_len_arr: np.array, 
    err_capacity_arr: np.array,
):
    size_key_len_arr = len(vkey_len_arr)
    # vector, same length as the length of vector key size,
    # each element of this vector is the false acceptance corresponding to the key size
    false_acceptance_times = [0] * size_key_len_arr
    total_imposter_trial = 0
    intra_sum = 0
    number_of_users = len(test_ebds.keys())

    for index_user in range(0, number_of_users - 1):
        for index_another_user in range(index_user + 1, number_of_users):
            total_imposter_trial+=1
            intra_hamming = calc_hamming_dictance(test_ebds[index_user], 
                                                    test_ebds[index_another_user])
            intra_sum += intra_hamming
            for index_key_size in range(0, size_key_len_arr):
                if intra_hamming <= err_capacity_arr[index_key_size]:
                    false_acceptance_times[index_key_size] += 1

    print('mean hamming distance when calc FRA: ', intra_sum / total_imposter_trial)
    return np.divide(false_acceptance_times, (total_imposter_trial * 1.0) / 100)

