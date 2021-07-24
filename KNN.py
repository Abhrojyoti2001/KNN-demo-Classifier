"""Nearest Neighbor Demo Classifier as KNN"""

# Author: Abhrojyoti Chatterjee
# Last update: 25th july on 2021

import numpy as np


class KNN:
    """
    Classifier implementing the k-nearest neighbors vote.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for Predictor queries.

    weights : {'uniform', 'distance'} default='uniform'
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are weighted equally.

        - 'distance' : weight points by the inverse of their distance. In this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

    Methods
    --------
    fit : Fit the k - nearest neighbors classifier from the training datasets.
        Parameters : x,y

    predict : Predict the class labels for the provided data.
        Parameters : x, return_probability
            return_probability : bool, default=False

    predict_probability : Return probability estimates for the test data.
        Parameters : x

    Warning
    --------
    1. If your training data samples is less than 5 you have to set n_neighbors, because KNN always set n_neighbors = 5 by default.

    2. If it is found that two neighbors, neighbor k and k+1, have identical distances but different labels, the results will depend on the ordering of the training data.

    Examples
    --------
    Case 1: without influence

    >>>x = [[0], [1], [2], [3]]
    >>>y = [0, 0, 1, 1]
    >>>from KNN import KNN
    >>>knn = KNN(n_neighbors=3)
    >>>knn.fit(x, y)
    >>>print(knn.predict([[1.1]]))
    [0]
    >>>print(knn.predict_probability([[0.9]]))
    {'0-item': array([0.66666667, 0.33333333])}

    Case 2: with influence

    >>>x = [[0], [1], [2], [3]]
    >>>y = [0, 0, 1, 1]
    >>>from KNN import KNN
    >>>knn = KNN(2,'distance')
    >>>knn.fit(x, y)
    >>>print(knn.predict([[1.1]]))
    [0]
    >>>print(knn.predict_probability([[1.9]]))
    {'0-item': dict_values([0.10000000000000009, 0.9900990099009901])}
    """

    def __init__(self, n_neighbors: int = 5, weights='uniform'):
        if type(n_neighbors) == int:
            if n_neighbors > 0:
                self.__n_neighbors = n_neighbors
            else:
                print("You should give positive integer number.")
        else:
            print("Unsupported data-type. n_neighbors not support in {} support only in int.".format(type(n_neighbors)))
        self.__weights = weights

    def fit(self, x, y):
        """
        Fit the k - nearest neighbors classifier from the training datasets.

        Parameters
        ----------
        x: {array - like, sparse matrix} of shape(n_samples, n_features) from Training data.
        y: {array - like, sparse matrix} of shape(n_samples, ) or (n_samples, n_outputs) Target values.
        """
        self.__X = np.array(x)
        self.__Y = np.array(y)

    def predict(self, x, return_probability=False):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features) Test samples.
        return_probability : bool, default False, return_probability function return probability when it is True.

        Returns
        ----------
        y : nd-array of shape (n_queries,) or (n_queries, n_outputs) Class labels for each data sample, if return_probability is False.
        p : nd-array of shape (n_queries,) or (n_queries, n_outputs) probability of each data sample, if return_probability is True.

        """
        arr_x = np.array(x)
        y = []
        p = {'0-item': None}
        for i in range(arr_x.shape[0]):
            index = []
            distances = self.__distance(arr_x[[i]])
            if self.__weights == 'distance':
                influence = {0: 0.0, 1: 0.0}
                loop_len = list(set(self.__Y))
                for j in range(len(loop_len)):
                    if loop_len[j] not in influence:
                        influence[loop_len[j]] = 0.0

                k_neighbors = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
                for k in k_neighbors[0:self.__n_neighbors]:
                    influence[self.__Y[k[0]]] += k[1]

                if return_probability is True:
                    for key in influence:
                        influence[key] /= sum(influence.values())

                    p['{}-item'.format(i)] = influence.values()
                else:
                    max_ = max(influence.values())
                    for key in influence:
                        if influence[key] == max_:
                            break
                    y.append(key)
            else:
                k_neighbors = sorted(list(enumerate(distances)), key=lambda x: x[1])
                for j in k_neighbors[0:self.__n_neighbors]:
                    index.append(j[0])

                if return_probability is True:
                    p['{}-item'.format(i)] = np.bincount(self.__Y[index]) / self.__n_neighbors
                else:
                    y.append(np.bincount(self.__Y[index]).argmax())

        if return_probability is True:
            return p
        else:
            return np.array(y)

    def __distance(self, x):
        """
        Finds the K - neighbors and distance of a point.

        Parameters
        ----------
        x : array - like of shape(1, n_features)

        Returns
        ----------
        distance_list array of shape(1, n_neighbors_dist). Array representing the n_neighbors of all points of x, return influence of n_neighbors if weights = distance.
        """
        distance = 0.0
        distance_list = []
        for i in range(self.__X.shape[0]):
            for j in range(self.__X.shape[1]):
                distance = distance + (self.__X[i][j] - x[0][j]) ** 2

            if self.__weights == 'distance':
                distance_list.append(1 / (distance ** 0.5))  # influence = 1/distance
            else:
                distance_list.append(distance ** 0.5)
            distance = 0.0

        return distance_list

    def predict_probability(self, x):
        """
        Return probability estimates for the test data x.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_features) Test samples.

        Returns
        -------
        p : nd-array of shape (n_queries, n_classes), or a list of n_outputs of such arrays if n_outputs > 1. The class probabilities of the input samples.
        """
        p = self.predict(x, True)

        return p
