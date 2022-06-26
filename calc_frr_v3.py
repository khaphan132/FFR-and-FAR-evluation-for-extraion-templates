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
    embeddings_data: object, vkey_len_arr: np.array, err_capacity_arr: np.array,
):
    size_key_len_arr = len(vkey_len_arr)

    # vector, same length as the length of vector key size,
    # each element of this vector is the false rejection corresponding to the key size
    false_rejection_times = [0] * size_key_len_arr
    total_genuine_trial = 0
    number_of_enrolled_vectors = 2
    intra_sum = 0

    for user in embeddings_data.keys():
        # print(user)
        enrolled_vectors, test_vectors = split_1_array_to_2_array(embeddings_data[user], number_of_enrolled_vectors, True)
        avg_enrolled_vector = get_binary_vector_of_real_vectors(enrolled_vectors)

        size_subset = 2
        for subset_test_vectors in itertools.combinations(test_vectors, size_subset):
            total_genuine_trial += 1
            avg_test_vector = get_binary_vector_of_real_vectors(subset_test_vectors)
            intra_hamming = calc_hamming_dictance(avg_enrolled_vector, avg_test_vector)
            intra_sum += intra_hamming

            for index_key_size in range(0, size_key_len_arr):
                if intra_hamming > err_capacity_arr[index_key_size]:
                    false_rejection_times[index_key_size] += 1

    print("mean hamming distance when calc FRR: ", intra_sum / total_genuine_trial)
    return np.divide(false_rejection_times, (total_genuine_trial * 1.0) / 100)
