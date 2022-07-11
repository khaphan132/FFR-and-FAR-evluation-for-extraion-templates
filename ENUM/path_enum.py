from enum import Enum

class PATH_ENUM(Enum):
    TIMIT_LIST_FOLDER = 'embeddings/vkyc/timit_list'
    TIMIT_LIST_TRAIN = 'embeddings/timit/timit_list/TIMIT_train.scp'
    TIMIT_LIST_TEST = 'embeddings/timit/timit_list/TIMIT_test.scp'
    TIMIT_CLASS_DICT_FILE = 'embeddings/timit/timit_list/TIMIT_labels.npy'
    TIMIT_EMBEDING_V41 = 'embeddings/timit/ebd/timit-v41'
    TIMIT_EMBEDING_V46 = 'embeddings/timit/ebd/timit-v46'
    TIMIT_EMBEDING_V48 = 'embeddings/timit/ebd/timit-v48'
    
    VKYC_LIST_FOLDER = 'embeddings/vkyc/vkyc_v2_list'
    VKYC_LIST_ALL = 'embeddings/vkyc/vkyc_v2_list/vkyc_all.scp'
    VKYC_LIST_TRAIN = 'embeddings/vkyc/vkyc_v2_list/vkyc_train.scp'
    VKYC_LIST_TEST = 'embeddings/vkyc/vkyc_v2_list/vkyc_test.scp'
    VKYC_CLASS_DICT_FILE = 'embeddings/vkyc/vkyc_v2_list/vkyc_labels.npy'
    VKYC_EMBEDING_V14 = 'embeddings/vkyc/ebd/vkyc-v14'
    VKYC_EMBEDING_V30 = 'embeddings/vkyc/ebd/vkyc-v30'
    VKYC_EMBEDING_V33 = 'embeddings/vkyc/ebd/vkyc-v33'