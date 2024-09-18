import numpy as np


class Node:
    def __init__(self, X=None):
        self.X = X                      # train partition
        self.feature = None             # feature split on
        self.value = None               # feature value split on   
        self.revenue = None             # posterior revenue
        self.left = None                # left child
        self.right = None               # right child
        self.X_test = None              # test paritition

class RiskAverseSPT:

    def __init__(self, teacher, min_samples=1, risk_proba=0.05):
        """
        parameters:
        teacher (object): an instantiation of a class that has a `predict_proba` method
            which returns a values in (0, 1)
        """
        
        self.teacher = teacher          # teacher model
        self.prices = []                # discrete price options
        self.risk_proba = risk_proba    # risk probability
        self.min_samples = min_samples  # min_samples 
        self.total_rev = 0              # predicted total revenue

    def _is_risk_averse(self, node):
        """
        Evaluates if the current split obeys the Value-at-Risk criteria.

        returns: 0 or 1
        """
        if self._revenue_increased(node):
            # new revenue posterior
            new_rev = node.left.revenue.sum(axis=0) + node.right.revenue.sum(axis=0)
            # old revneue posterior
            old_rev = node.revenue.sum(axis=0)
            # prob that revenue decreases
            p =  new_rev - old_rev  
            # return risk criteria boolean
            return sum(p < 0) / len(p) < self.risk_proba
        else:
            return False

    def _get_expected_revenue(self, X, price):
        """
        Compute expected revenue given covariates X and price.
        """
        # predict revenue posterior
        rev = self.teacher.predict_proba(X, price) * price
        # return posterior revenue
        return rev

    def _total_rev(self, revenue):
        """
        Total expected revenue in partition.
        """
        if revenue is not None:
            return revenue.sum(axis=0).mean()
        else:
            return 0

    def _update_partition(self, node, X1, X2, rev1, rev2, p1, p2, feature, value):
        """
        Update node attributes if partition gives higher total revenue.
        """
        # aggregate posteriors
        old_rev = self._total_rev(node.left.revenue) \
            + self._total_rev(node.right.revenue)
        # get total revenue
        new_rev = self._total_rev(rev1) + self._total_rev(rev2)
        if new_rev > old_rev:
            # update split params
            self._update_params(node, feature, value)
            # update left partition
            self._update_attrs(node.left, X1, p1, rev1)
            # update right partition
            self._update_attrs(node.right, X2, p2, rev2)

    def _split_data(self, X, feature, value):
        """
        Split data using split parameters.
        """
        # get boolean
        t = X[:, feature] >= value
        # split data
        X1, X2 =  X[t], X[~t]
        # return split
        return X1, X2

    def _update_best_split(self, node, value, feature):
        """
        For a feature index j and feature value update nodes.
        """
        # split data
        X1, X2 = self._split_data(node.X, feature, value)
        # get left revenue posterior rev1 for best price p1
        p1, rev1 = self._get_best_price(X1)
        # get right revenue posterior rev2 for best price p2
        p2, rev2 = self._get_best_price(X2)
        # update best parition node
        self._update_partition(
            node, X1, X2, rev1, rev2, p1, p2, feature, value
        )

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
        x = (x[1:] + x[:-1]) / 2
        # trim cuts near the edges
        return x[self.min_samples: -self.min_samples]

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
        for j in range(node.X.shape[1]):
            # get all split values
            values = self._get_feature_splits(node, j)
            # iterate values
            for value in values:
                # update best split
                self._update_best_split(node, value, j)

    def _revenue_increased(self, node):
        """
        Check that split occured
        """
        return not (
            (node.left.revenue is None) | (node.right.revenue is None)
        )

    def _split(self, node):
        """
        Split node into left and right.
        """
        # add left and right
        self._add_children(node)
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

    def _add_children(self, node):
        """
        Create node with left and right children.
        """
        node.left = Node()
        node.right = Node()

        return node

    def _get_best_price(self, X):
        """
        Find best price for max rev.
        """
        # get revenues
        revs = [self.teacher.predict_proba(X, price) * price for price in self.prices]
        # get idx of best price
        idx = np.argmax([self._total_rev(rev) for rev in revs])
        # return price, revenue
        return self.prices[idx], revs[idx]

    def _update_attrs(self, node, X, price, revenue):
        """
        Update node attributes.
        """
        # update attributes
        node.X = X
        node.price = price
        node.revenue = revenue

    def _update_params(self, node, feature, value):
        """
        Update split parameters.
        """
        node.feature = feature
        node.value = value

    def _preprocess(self, X, prices):
        """
        Store variables and convert data to numpy.
        """
        # store length of training set
        self.n_train = len(X)
        # store prices
        self.prices = prices
        # safely convert data to numpy
        X = np.asarray(X)
        return X

    def fit(self, X, prices):
        """
        Fit the Student Prescription Tree.
        """
        # prepare data
        X = self._preprocess(X, prices)
        # create root node
        self.root_node = Node()
        # get best price for partition
        price, revenue = self._get_best_price(X)
        # update attribtues
        self._update_attrs(self.root_node, X, price, revenue)
        # split node
        self._split(self.root_node)

    def _transform(self, X, node):
        """
        Recursive function for self.transform()
        """
        # update node test data
        node.X_test = X
        # get split params
        feature, value = node.feature, node.value
        # split data
        X1, X2 = self._split_data(X, feature, value)
        # if not leaf node
        if not self._is_leaf(node):
            # recursively split
            self._transform(X1, node.left)
            self._transform(X2, node.right)

    def transform(self, X):
        """
        Parition X into learned decision tree.
        """
        # get root
        node = self.root_node
        # safely convert data to numpy
        X = np.asarray(X)
        # transform recursivles
        self._transform(X, node)

    def _is_leaf(self, node):
        """
        Check if node is a leaf node.
        """
        return (node.left is None) | (node.right is None)

    def _expected_revenue(self, node):
        """
        Recursive function to add up revenue in leaf nodes.
        """
        # expected revenue
        rev = node.revenue.mean(axis=1).sum()
        # if leave node add revenue
        if self._is_leaf(node):
            # add revenue
            self.total_rev += rev
        else:
            # recursive left
            self._expected_revenue(node.left)
            # recursive right
            self._expected_revenue(node.right)

    def expected_revenue(self):
        """
        Add up revenue in leaf nodes.
        """
        # reset total revenue
        self.total_rev = 0
        # add rev in leaf nodes
        self._expected_revenue(self.root_node)
        # compute average revenue
        avg_rev = self.total_rev / self.n_train
        # return total and avg revenue
        return self.total_rev, avg_rev

