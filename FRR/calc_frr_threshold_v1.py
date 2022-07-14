import numpy as np

from utils import calc_hamming_dictance


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
    threshold_arr: np.array,
):
    len_threshold_arr = len(threshold_arr)
    false_rejection_times = [0] * len_threshold_arr
    total_genuine_trial = 0
    intra_sum = 0

    for user in enroll_ebds.keys():
        # print(user)
        for test_ebd in test_ebds[user]:
            total_genuine_trial += 1
            intra_hamming = calc_hamming_dictance(enroll_ebds[user], test_ebd)
            intra_sum += intra_hamming

            for i in range(0, len_threshold_arr):
                if intra_hamming > threshold_arr[i]:
                    false_rejection_times[i] += 1
    print("\ntotal_genuine_trial & sum_intra_hamming: ", total_genuine_trial, intra_sum)
    print("mean hamming distance when calc FRR: ", intra_sum / total_genuine_trial)
    return np.divide(false_rejection_times, (total_genuine_trial * 1.0) / 100)
