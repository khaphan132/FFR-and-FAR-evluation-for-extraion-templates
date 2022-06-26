from enum import Enum

class PATH_ENUM(Enum):
    TIMIT_LIST_FOLDER = 'embeddings/vkyc/timit_list'
    TIMIT_LIST_TRAIN = 'embeddings/timit/timit_list/TIMIT_train.scp'
    TIMIT_LIST_TEST = 'embeddings/timit/timit_list/TIMIT_test.scp'
    TIMIT_EMBEDING_NOTRAIN_V41_HM2 = 'embeddings/timit/ebd/timit_notrain_v41_hm2'
    TIMIT_CLASS_DICT_FILE = 'embeddings/timit/timit_list/TIMIT_labels.npy'
    
    VKYC_LIST_FOLDER = 'embeddings/vkyc/vkyc_v2_list'
    VKYC_LIST_TRAIN = 'embeddings/vkyc/vkyc_v2_list/vkyc_train.scp'
    VKYC_LIST_TEST = 'embeddings/vkyc/vkyc_v2_list/vkyc_test.scp'
    VKYC_CLASS_DICT_FILE = 'embeddings/vkyc/vkyc_v2_list/vkyc_labels.npy'
    VKYC_EMBEDING_V9_HM2 = 'embeddings/vkyc/ebd_vkyc_v9_hm2'
    VKYC_EMBEDING_V14_EC2 = 'embeddings/vkyc/ebd_vykc_v14_ec2'