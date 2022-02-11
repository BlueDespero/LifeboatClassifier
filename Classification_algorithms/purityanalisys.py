import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def purity_summary(data, purity_function):
    col_names = []
    purity_measures = []
    for name, data in data.iteritems():
        col_names.append(name)
        purity_measures.append(purity_function(data.value_counts()))

    plot_results(col_names, purity_measures)


if __name__ == "__main__":
    full_data = pd.read_csv("../data/train.csv")
    purity_summary(full_data, pm.gini)
