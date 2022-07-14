import numpy as np

from utils import calc_hamming_dictance


def calcFARs(
    test_ebds: dict, threshold_arr: np.array,
):
    len_threshold_arr = len(threshold_arr)
    false_acceptance_times = [0] * len_threshold_arr
    total_imposter_trial: np.int64 = 0
    intra_sum: np.double = 0

    list_users = list(test_ebds.keys())
    number_of_users = len(list_users)

    for index_user in range(0, number_of_users - 1):
        for user_ebd in test_ebds[list_users[index_user]]:
            for index_another_user in range(index_user + 1, number_of_users):
                for another_user_ebd in test_ebds[list_users[index_another_user]]:
                    total_imposter_trial += 1
                    intra_hamming = calc_hamming_dictance(user_ebd, another_user_ebd)
                    intra_sum += intra_hamming

                    for index_threshold in range(0, len_threshold_arr):
                        if intra_hamming <= threshold_arr[index_threshold]:
                            false_acceptance_times[index_threshold] += 1

    print("\ntotal_imposter_trial & sum_intra_hamming", total_imposter_trial, intra_sum)
    print("mean hamming distance when calc FAR: ", intra_sum / total_imposter_trial)
    return np.divide(false_acceptance_times, (total_imposter_trial * 1.0) / 100)

