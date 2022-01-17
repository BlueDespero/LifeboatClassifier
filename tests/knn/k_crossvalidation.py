import pickle

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

results_set = []

for key, encoder in tqdm.tqdm(encoders.items()):
    for k in range(1, 101):
        train_x, train_y = get_data("../../data/train.csv", encoder=encoder)
        result = cross_validation(train_x, train_y, classifier=KNN(), folds=10, k=k, encoder=encoder)
        results_set.append(result)

    with open("../../Classification_algorithms/Predictions/knn_crossval_%s_results.pickle" % key, 'wb') as handle:
        pickle.dump(results_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    results_set = []
