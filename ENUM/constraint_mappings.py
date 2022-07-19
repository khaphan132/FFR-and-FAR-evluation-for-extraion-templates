from sympy import get_contraction_structure


timit_constraint_mappings = {
    "timit-v41": "Square constraint",
    "timit-v46": "Triplet constraint",
    "timit-v48": "Modulus constraint",    
}

vkyc_constraint_mappings = {
    "vkyc-v14": "Square constraint",
    "vkyc-v30": "Triplet constraint",
    "vkyc-v33": "Modulus constraint",    
}

def get_constraint_string(dataset: str):
    if (dataset.find("vkyc") != -1):
        return vkyc_constraint_mappings[dataset]
    if (dataset.find("timit") != -1):
        return timit_constraint_mappings[dataset]
    return ""
