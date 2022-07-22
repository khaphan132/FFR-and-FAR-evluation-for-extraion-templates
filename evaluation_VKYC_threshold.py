import os
import numpy as np
from ENUM.constraint_mappings import get_constraint_string
from ENUM.error_toleration_enum import ERROR_TOLERATION_ENUM
from FRR.calc_frr_threshold import calcFRRs
from FAR.calc_far_threshold import calcFARs
from data_io import ReadList, getEmbeddingsData, write_report_threshold
from ENUM.path_enum import PATH_ENUM, get_VKYC_path_ebds
from utils import (
    calc_average_of_arrays,
    get_key_len_and_error_capacity_by_size,
    saveChart2Line,
    split_enroll_and_test_embeddings,
)


# load embeddings from test folders
# store it as following dictionary
def prepare_list_wav():
    """  Loading label dictionary """
    lab_dict = np.load(PATH_ENUM.VKYC_CLASS_DICT_FILE.value, allow_pickle=True).item()

    """  test list """
    wav_lst_te = ReadList(PATH_ENUM.VKYC_LIST_EVAL.value)
    return lab_dict, wav_lst_te


def execute_evaluate(lab_dict, wav_lst_te, path_ebd):
    """  embedding set that perform evaluating """
    embeddings = {}
    embeddings = getEmbeddingsData(wav_lst_te, lab_dict, path_ebd)

    """ split embeddings to 2 sets: enroll & test """
    number_of_enroll_embeddings_per_user = 5
    enroll_ebds, test_ebds = split_enroll_and_test_embeddings(embeddings, number_of_enroll_embeddings_per_user)

    print("--- test case ---")
    print("evaulate with threshold")
    print("embedding set: " + path_ebd.split("/")[-1])

    """  calculate FRR & FAR for extraction templates """
    threshold_arr = ERROR_TOLERATION_ENUM.THRESHOLD_ARR.value
    FRRs = calcFRRs(enroll_ebds, test_ebds, threshold_arr)
    FARs = calcFARs(test_ebds, threshold_arr)

    # EERs = calc_average_of_arrays([FRRs, FARs]) 
    EERs = np.abs(np.subtract(FRRs, FARs))
    eer_min = np.min(EERs)
    eer_min_index = np.argmin(EERs)
    print(
        "FAR, FRR, EER_index and threshold: ", FRRs[eer_min_index], FARs[eer_min_index], eer_min_index, threshold_arr[eer_min_index]
    )

    """  save result """
    output_folder = "output_VKYC/output_" + path_ebd.split("/")[-1] + "_" + "threshold/"
    print("output folder: " + output_folder)
    try:
        os.stat(output_folder)
    except:
        os.mkdir(output_folder)

    write_report_threshold(
        FRRs[eer_min_index], 
        FARs[eer_min_index], 
        np.mean([FRRs[eer_min_index], FARs[eer_min_index]]),
        threshold_arr[eer_min_index], 
        output_folder
    )
    np.savetxt(output_folder + "frr.txt", FRRs, delimiter=",")
    np.savetxt(output_folder + "far.txt", FARs, delimiter=",")
    saveChart2Line(
        [threshold_arr, FRRs],
        "FRR",
        [threshold_arr, FARs],
        "FAR",
        "Hamming distance threshold",
        "error rate (%)",
        "FAR & FRR with VKYC - " + get_constraint_string(path_ebd.split("/")[-1]),
        output_folder + "FRR_and_FAR_with_VKYC_" + get_constraint_string(path_ebd.split("/")[-1]).replace(" ", "_") + ".png",
    )

    print("\n\n===================== done ===========================\n\n")


lab_dict, wav_lst_te = prepare_list_wav()
for path_ebd in get_VKYC_path_ebds():
    execute_evaluate(lab_dict, wav_lst_te, path_ebd)

