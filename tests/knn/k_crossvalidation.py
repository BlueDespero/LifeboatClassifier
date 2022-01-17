import itertools
import pickle
import warnings

import tqdm
from category_encoders import BackwardDifferenceEncoder, BaseNEncoder, BinaryEncoder, CatBoostEncoder, CountEncoder, \
    GLMMEncoder, HashingEncoder, HelmertEncoder, JamesSteinEncoder, LeaveOneOutEncoder, MEstimateEncoder, OneHotEncoder, \
    OrdinalEncoder, SumEncoder, PolynomialEncoder, TargetEncoder, WOEEncoder

from Classification_algorithms.K_Nearest_Neighbors.k_nearest_neighbors import KNN
from tests.utils import cross_validation
from utils.data_preprocessing import get_data

encoders = {
    "BackwardDifferenceEncoder": BackwardDifferenceEncoder,
    "BaseNEncoder": BaseNEncoder,
    "BinaryEncoder": BinaryEncoder,
    "CatBoostEncoder": CatBoostEncoder,
    "CountEncoder": CountEncoder,
    "GLMMEncoder": GLMMEncoder,
    "HashingEncoder": HashingEncoder,
    "HelmertEncoder": HelmertEncoder,
    "JamesSteinEncoder": JamesSteinEncoder,
    "LeaveOneOutEncoder": LeaveOneOutEncoder,
    "MEstimateEncoder": MEstimateEncoder,
    "OneHotEncoder": OneHotEncoder,
    "OrdinalEncoder": OrdinalEncoder,
    "SumEncoder": SumEncoder,
    "PolynomialEncoder": PolynomialEncoder,
    "TargetEncoder": TargetEncoder,
    "WOEEncoder": WOEEncoder
}

distances = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "jensenshannon",
             "mahalanobis", "matching", "minkowski", "russellrao", "seuclidean", "sqeuclidean", "wminkowski"]

results_set = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for encoder, distance in tqdm.tqdm(list(itertools.product(encoders.items(), distances))):
        key, encoder = encoder
        for k in range(1, 101):
            train_x, train_y = get_data("../../data/train.csv", encoder=encoder)
            result = cross_validation(train_x, train_y, classifier=KNN(), folds=10, k=k, encoder=encoder, distance=distance)
            results_set.append(result)

        with open("../../Classification_algorithms/Predictions/knn_crossval_%s_%s_results.pickle" % (key, distance),
                  'wb') as handle:
            pickle.dump(results_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

        results_set = []
