import matplotlib.pyplot as plt
import numpy as np

import Classification_algorithms.Decision_Tree.purity_measures as pm


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


def gini_summary(data):
    col_names = []
    purity_measures = []
    for name, data in data.iteritems():
        col_names.append(name)
        purity_measures.append(pm.gini(data.value_counts()))

    plot_results(col_names, purity_measures)


def entropy_summary(data):
    col_names = []
    purity_measures = []
    for name, data in data.iteritems():
        col_names.append(name)
        purity_measures.append(pm.entropy(data.value_counts()))

    plot_results(col_names, purity_measures)


def mer_summary(data):
    col_names = []
    purity_measures = []
    for name, data in data.iteritems():
        col_names.append(name)
        purity_measures.append(pm.mean_err_rate(data.value_counts()))

    plot_results(col_names, purity_measures)
