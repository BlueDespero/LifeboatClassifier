import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import cycle

import Classification_algorithms.Decision_Tree.purity_measures as pm

def purity_summary(data, purity_functions):
    result_dataframe = pd.DataFrame()

    for p_function, purity_name in purity_functions:
        col_names = []
        purity_measures = []
        for name, col_data in data.iteritems():
            col_names.append(name)
            purity_measures.append(p_function(col_data.value_counts()))

        p_dataframe = pd.DataFrame(list(zip(cycle([purity_name]), purity_measures, col_names)),
                                   columns=["purity_name", "purity_measure", "feature"])
        result_dataframe = pd.concat([result_dataframe, p_dataframe])


if __name__ == "__main__":
    full_data = pd.read_csv("../data/train.csv")
    purity_summary(full_data, [(pm.gini, "gini"), (pm.entropy, "entropy")])
