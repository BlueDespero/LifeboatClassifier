from Classification_algorithms.K_Nearest_Neighbors.k_nearest_neighbors import KNN
from tests.utils import cross_validation
from utils.data_preprocessing import get_data

train = get_data("../../data/train_num.csv")
test = get_data("../../data/test_num.csv")

train_x = train.iloc[:, 2:]
train_y = train.iloc[:, 1]

result = cross_validation(train_x, train_y, classifier=KNN(), folds=10, k=5)
print(result)
