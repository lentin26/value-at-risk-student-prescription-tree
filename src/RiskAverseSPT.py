import numpy as np


class Node:
    def __init__(self, X, y):
        # data points, feature index and value for split
        self.X = X
        self.y = y
        self.feature = None # feature split on
        self.value = None   # feature value split on
        self.revenue = 0
        # link to left and right children
        self.left = None
        self.right = None


class RiskAverseSPT:

    def __init__(self, teacher, risk_proba=0.05):
        """
        parameters:
        teacher (object): an instantiation of a class that has a `predict_proba` method
            which returns a values in (0, 1)
        """
        # teacher model
        self.teacher = teacher
        # discrete price options
        self.prices = []
        # risk probability
        self.risk_proba = risk_proba

    def _is_risk_averse(self, node):
        """
        Evaluates if the current split obeys the Value-at-Risk criteria.

        returns: 0 or 1
        """
        # compute risk probability
        p = sum(node.revenue - (node.left.revenue + node.right.revenue))
        # evaluate if it is above the min threshold
        return int(p > self.risk_proba)

    def _get_expected_revenue(self, X, price):
        """
        Compute expected revenue given covariates X and price.
        """
        # reshape price
        price = np.repeat(price, X.shape[0]).reshape(-1, 1)
        # concatenate price and covariates
        X_full = np.append([price, X])
        # predict revenue posterior
        rev = self.teacher.predict_proba(X_full) * price
        # return posterior revenue
        return rev

    def _update_node(self, node, X, revenue, price, value, feature):
        """
        Update node attributes.
        """
        if sum(revenue) > sum(node.revenue):
            # update best revenue
            node.revenue = revenue
            # update best price
            node.price = price
            # update feature value
            node.value = value
            # update feature index
            node.feature = feature
            # update data
            node.X = X

    def _update_best_split(self, node, value, j):
        """
        For a feature index j and feature value update nodes.
        """
        # get split boolean
        t = node.X[:, j] >= value
        # split data
        X1, X2 =  node.X[t], node.X[~t]
        # iterate prices
        for price in self.prices:
            # get left revenue posterir
            R1 = self._get_expected_revenue(X1, price)
            # get right revenue posterior
            R2 = self._get_expected_revenue(X2, price)
            # update left node
            self._update_node(self, node.left, X1, R1, price, value, j)
            # update right node
            self._update_node(self, node.right, X2, R2, price, value, j)


    def _get_feature_splits(self, node, j):
        """
        Sort feature values and return midpoints.

        parameters:
        j (int): positional index of feature in X

        return: List of sorted features
        """
        # get unique features values and sort
        x = np.sort(np.unique(node.X[:, j])) 
        # return value midpoits
        return (x[1:] + x[:-1]) / 2

    def _remove_children(self, node):
        """
        Remove children from node.
        """
        # erase left
        node.left = None
        # erase right
        node.right= None

    def _split_node(self, node):
        """
        Iterate feature values and update optimal split.
        """
        # iterate through features
        for j in node.X.shape[1]:
            # get all split values
            values = self._get_feature_splits(node, j)
            # iterate values
            for value in values:
                # update best split
                self._update_best_split(node, value, j)

    def _split(self, node):
        """
        Split node into left and right.
        """
        # split node into left and right
        self._split_node(node)
        # if risk-averse, recursively split
        if self._is_risk_averse(node):
            # split left
            self._split(node.left)
            # split right
            self._split(node.right)
        else:
            # otherwise, remove left and right
            self._remove_children(node)

    def fit(self, X, y):
        """
        Fit the Student Prescription Tree.
        """
        # create root node
        self.root_node = Node(X, y)
        # split node
        self._split(self.root_node)


