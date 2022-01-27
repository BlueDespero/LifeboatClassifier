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
    #    "CatBoostEncoder": CatBoostEncoder,
    "CountEncoder": CountEncoder,
    #    "GLMMEncoder": GLMMEncoder,
    "HashingEncoder": HashingEncoder,
    "HelmertEncoder": HelmertEncoder,
    #    "JamesSteinEncoder": JamesSteinEncoder,
    #    "LeaveOneOutEncoder": LeaveOneOutEncoder,
    #    "MEstimateEncoder": MEstimateEncoder,
    "OneHotEncoder": OneHotEncoder,
    "OrdinalEncoder": OrdinalEncoder,
    "SumEncoder": SumEncoder,
    "PolynomialEncoder": PolynomialEncoder
    #    "TargetEncoder": TargetEncoder,
    #    "WOEEncoder": WOEEncoder
}

# distances = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "jensenshannon",
#             "mahalanobis", "matching", "minkowski", "russellrao", "seuclidean", "sqeuclidean", "wminkowski"]

distances = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "jensenshannon",
             "matching", "minkowski", "russellrao", "seuclidean", "sqeuclidean"]

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for encoder, distance in tqdm.tqdm(list(itertools.product(encoders.items(), distances))):
            try:

                key, encoder = encoder
                train_x, train_y = get_data("../../data/train.csv", encoder=encoder)
                results = cross_validation(train_x, train_y, classifier=KNN(), folds=10, k=list(range(1, 101)),
                                           encoder=encoder, distance=distance)

                with open("../../Classification_algorithms/Predictions/knn_crossval_%s_%s_results.pickle" % (key, distance),
                          'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                results_set = []
            except RuntimeError:
                continue
