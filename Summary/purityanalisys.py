import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Classification_algorithms.Decision_Tree import purity_measures as pm

all_data = pd.read_csv("../data/train.csv")


def plot_results(col_names, purity_measures):
    purity_measures, col_names = zip(*sorted(zip(purity_measures, col_names), reverse=True))
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    y_pos = np.arange(len(col_names))

    ax.barh(y_pos, purity_measures, align='center')
    ax.set_yticks(y_pos, labels=col_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.show()


def gini_summary():
    col_names = []
    purity_measures = []
    for name, data in all_data.iteritems():
        col_names.append(name)
        purity_measures.append(pm.gini(data.value_counts()))

    plot_results(col_names, purity_measures)


def entropy_summary():
    col_names = []
    purity_measures = []
    for name, data in all_data.iteritems():
        col_names.append(name)
        purity_measures.append(pm.entropy(data.value_counts()))

    plot_results(col_names, purity_measures)


def mer_summary():
    col_names = []
    purity_measures = []
    for name, data in all_data.iteritems():
        col_names.append(name)
        purity_measures.append(pm.mean_err_rate(data.value_counts()))

    plot_results(col_names, purity_measures)
