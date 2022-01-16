class AbstractClassifier:

    def __init__(self):
        self.train_x = None
        self.train_y = None

    def __str__(self):
        raise NotImplementedError

    def train(self, train_x, train_y, **kwargs):
        self.train_x = train_x
        self.train_y = train_y

    def classify(self, x, **kwargs):
        raise NotImplementedError
