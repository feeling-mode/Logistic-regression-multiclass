import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pylab as plt


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
    
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return self.activation(self.net_input(X))

# _____________________________________________________________________________________________________________

class Multi_classifier: 
    
    def __init__(self, lrgd0, lrgd1, lrgd2):
        self.lrgd0=lrgd0
        self.lrgd1=lrgd1
        self.lrgd2=lrgd2
    
    def _predict_probability(self, X):
        p0 = self.lrgd0.activation(self.lrgd0.net_input(X))
        p1 = self.lrgd1.activation(self.lrgd1.net_input(X))
        p2 = self.lrgd2.activation(self.lrgd2.net_input(X))
        H = np.matrix([p0,p1,p2]).T
        return H
    
    def _predict_class(self, X):
        H = self._predict_probability(X)
        h = np.ones(len(H))
        h = np.where(H==np.amax(H, 1))[1] #tylko indeks(==klasa)
        return h
    
    def predict(self, X):
        return self._predict_class(X)
       
# _____________________________________________________________________________________________________________

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # konfiguruje generator znaczników i mapę kolorów
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # rysuje wykres powierzchni decyzyjnej
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # rysuje wykres wszystkich próbek
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')

# _____________________________________________________________________________________________________________

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)
    
    # subset 0: 0=>1; 1,2=>-1
    X_train_00_subset = X_train.copy()
    y_train_00_subset = y_train.copy()
    # w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa)  
    y_train_00_subset[(y_train_00_subset != 0)] = -1
    y_train_00_subset[(y_train_00_subset == 0 )] = 1
    y_train_00_subset[(y_train_00_subset == -1)] = 0
    lrgd0 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd0.fit(X_train_00_subset, y_train_00_subset)
    
    # subset 2: 2=>1; 0,1=>-1
    X_train_02_subset = X_train.copy()
    y_train_02_subset = y_train.copy()
    # w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa) 
    y_train_02_subset[(y_train_02_subset != 2 )] = 0
    y_train_02_subset[(y_train_02_subset == 2 )] = 1
    lrgd2 = LogisticRegressionGD(eta=0.05, n_iter=5000, random_state=1)
    lrgd2.fit(X_train_02_subset, y_train_02_subset)
    
        # subset 2: 2=>1; 0,1=>-1
    X_train_01_subset = X_train.copy()
    y_train_01_subset = y_train.copy()
    # w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa) 
    y_train_01_subset[(y_train_01_subset != 1 )] = 0
    # y_train_02_subset[(y_train_02_subset == 2 )] = 1
    lrgd1 = LogisticRegressionGD(eta=0.006, n_iter=2000, random_state=1)
    lrgd1.fit(X_train_01_subset, y_train_01_subset)

    # _____________________________________________________________________________________________________________

    
    multi_class = Multi_classifier(lrgd0, lrgd1, lrgd2)
    #print('\npredicted probabilities for classes 0, 1, 2: \n', np.round(predict_probability( X, lrgd0, lrgd2, lrgd1),3))
    
    plot_decision_regions(X=X, y=y, classifier=multi_class)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
