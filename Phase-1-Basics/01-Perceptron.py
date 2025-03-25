import numpy as np

def unit_step_fn(x):
    return np.where(x > 0, 1, 0)

class Perceptron:

    def __init__(self, learning_rate = 0.01, n_iter = 100) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.activation_fn = unit_step_fn
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        num_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        # learn weights
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_fn(linear_output)

                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):

        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_fn(linear_output)

        return y_predicted


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        return np.sum(y_pred == y_true) / len(y_true)

    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    p = Perceptron(learning_rate=0.01, n_iter=1000)
    p.fit(X_train, y_train)

    predictions = p.predict(X_test)

    print("Perceptron classification accuracy: ", accuracy(y_test, predictions))


    


