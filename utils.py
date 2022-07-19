import random
import numpy as np
import matplotlib.pyplot as plt
from ENUM.error_toleration_enum import ERROR_TOLERATION_ENUM

def split_1_array_to_2_array(arr: np.array, index_split: np.number, isShuffle: bool = False):
    if (index_split >= len(arr)):
        return ([], arr)
    if (isShuffle):
        random.shuffle(arr)
    split_arrs = np.split(arr, [index_split])
    return (split_arrs[0], split_arrs[1])


def get_binary_vector_of_real_vectors(float_vectors: np.array):
    return np.sign(calc_average_of_arrays(float_vectors))


def get_test_vectors_of_user(float_test_vectors: np.array):
    return np.sign(float_test_vectors)


def calc_average_of_arrays(list_arrays: np.array):
    result = list_arrays[0]
    len_list = len(list_arrays)
    for i in range(1, len_list):
        result = np.add(result, list_arrays[i])
    return np.divide(result, len_list)


def calc_hamming_dictance(a: np.array, b: np.array):
    return (a.shape[0] - dot_product(a, b)) / 2


def dot_product(a: np.array, b: np.array):
    return (a * b).sum()


def get_key_len_and_error_capacity_by_size(size: np.number):
    if (size == 255):
        return ERROR_TOLERATION_ENUM.KEY_LEN_255.value, ERROR_TOLERATION_ENUM.ERROR_CAPACITY_255.value
    if size == 511:
        return ERROR_TOLERATION_ENUM.KEY_LEN_511.value, ERROR_TOLERATION_ENUM.ERROR_CAPACITY_511.value
    if size == 1023:
        return ERROR_TOLERATION_ENUM.KEY_LEN_1023.value, ERROR_TOLERATION_ENUM.ERROR_CAPACITY_1023.value

def split_enroll_and_test_embeddings(embeddings: object, number_of_enroll_embeddings_per_user: int):
    enroll_ebds, test_ebds = {}, {}
    for user in embeddings.keys():
        enrolled_vectors, test_vectors = split_1_array_to_2_array(
                                        embeddings[user], 
                                        number_of_enroll_embeddings_per_user, 
                                        False)
        if (len(enrolled_vectors) != 0):
            enroll_ebds[user] = get_binary_vector_of_real_vectors(enrolled_vectors)
        test_ebds[user] = np.sign(test_vectors)
    return (enroll_ebds, test_ebds)

def saveChart2Line(
    line1: np.array,
    line1_label: str,
    line2: np.array,
    line2_label: str,
    x_label: str,
    y_label: str,
    title: str,
    path_export_graph: str = "output/frr-far.png",
):
    plt.clf()
    # plotting the line 1 points
    line1_color = '#4699b0'
    plt.plot(line1[0], line1[1], label=line1_label, color=line1_color)
    plt.scatter(line1[0], line1[1], fc='none', ec=line1_color)

    # plotting the line 2 points
    line2_color = '#f26135'
    plt.plot(line2[0], line2[1], label=line2_label, color=line2_color)
    plt.scatter(line2[0], line2[1], marker='*', fc='none', ec=line2_color)
    

    # naming the x axis
    plt.xlabel(x_label)
    # naming the y axis
    plt.ylabel(y_label)
    # giving a title to my graph
    plt.title(title)

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    # plt.show()
    # w = 640
    # h = 480
    plt.savefig(path_export_graph, dpi=1000)
