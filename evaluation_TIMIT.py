import os
import numpy as np
from FRR.calc_frr_v5 import calcFRRs
from FAR.calc_far_v6 import calcFARs
from data_io import ReadList, getEmbeddingsData, write_report_key_len, write_report_threshold
from ENUM.path_enum import PATH_ENUM, get_TIMIT_path_ebds
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
    lab_dict = np.load(PATH_ENUM.TIMIT_CLASS_DICT_FILE.value, allow_pickle=True).item()

    """  test list """
    wav_lst_te = ReadList(PATH_ENUM.TIMIT_LIST_TEST.value)
    return lab_dict, wav_lst_te


def execute_evaluate(lab_dict, wav_lst_te, path_ebd, key_len=1023):
    """  embedding set that perform evaluating """
    embeddings = {}
    embeddings = getEmbeddingsData(wav_lst_te, lab_dict, path_ebd)

    """ split embeddings to 2 sets: enroll & test """
    number_of_enroll_embeddings_per_user = 2
    enroll_ebds, test_ebds = split_enroll_and_test_embeddings(embeddings, number_of_enroll_embeddings_per_user)

    """  get vkey_len_arr, verr_capacity_arr """
    vkey_len_arr, verr_capacity_arr = get_key_len_and_error_capacity_by_size(key_len)  # codeword key_size bits

    print("--- test case ---")
    print("key_len: ", key_len)
    print("embedding set: " + path_ebd.split("/")[-1])

    """  calculate FRR & FAR for extraction templates """
    FRRs = calcFRRs(enroll_ebds, test_ebds, vkey_len_arr, verr_capacity_arr)
    FARs = calcFARs(test_ebds, vkey_len_arr, verr_capacity_arr)

    EERs = calc_average_of_arrays([FRRs, FARs])
    eer_min = np.min(EERs)
    eer_min_index = np.argmin(EERs)
    print(
        "FAR, FRR, EER and key_size, err_capacity: ",
        FRRs[eer_min_index],
        FARs[eer_min_index],
        eer_min,
        vkey_len_arr[eer_min_index],
        verr_capacity_arr[eer_min_index],
    )

    """  save result """
    output_folder = "output_TIMIT/output_" + path_ebd.split("/")[-1] + "_" + str(key_len) + "/"
    print("output folder: " + output_folder)
    try:
        os.stat(output_folder)
    except:
        os.mkdir(output_folder)

    write_report_key_len(
        FRRs[eer_min_index],
        FARs[eer_min_index],
        eer_min,
        vkey_len_arr[eer_min_index],
        verr_capacity_arr[eer_min_index],
        output_folder,
    )
    np.savetxt(output_folder + "frr.txt", FRRs, delimiter=",")
    np.savetxt(output_folder + "far.txt", FARs, delimiter=",")
    saveChart2Line(
        [vkey_len_arr, FRRs],
        "FRR",
        [vkey_len_arr, FARs],
        "FAR",
        "key size",
        "error rate (%)",
        path_ebd.split("/")[-1] + " - " + str(key_len) + "-bit codewords - FAR & FRR",
        output_folder + "frr-far.png",
    )

    print("\n\n===================== done ===========================\n\n")


lab_dict, wav_lst_te = prepare_list_wav()
for path_ebd in get_TIMIT_path_ebds():
    for key_len in [255, 511, 1023]:
        execute_evaluate(lab_dict, wav_lst_te, path_ebd, key_len)
