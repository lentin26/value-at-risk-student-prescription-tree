import numpy as np
import pandas as pd

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

    def __init__(self, teacher, prices, min_samples=1, risk_proba=0.05):
        """
        parameters:
        teacher (object): an instantiation of a class that has a `predict_proba` method
            which returns a values in (0, 1)
        """
        
        self.teacher = teacher          # teacher model
        self.prices = prices            # discrete price options
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
            # compute risk
            node.risk = (p < 0).sum() / len(p) 
            # return risk criteria boolean
            return node.risk < self.risk_proba
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

    def _update_partition(self, node, idx1, idx2, rev1, rev2, p1, p2, feature, value):
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
            self._update_attrs(node.left, idx1, p1, rev1)
            # update right partition
            self._update_attrs(node.right, idx2, p2, rev2)

    def _split_data(self, X, idx, feature, value):
        """
        Split data using split parameters.
        """
        # get boolean
        t = X[idx, feature] >= value
        # split data
        idx1, idx2 =  idx[t], idx[~t]
        # return split
        return idx1, idx2

    def _update_best_split(self, X, node, value, feature):
        """
        For a feature index j and feature value update nodes.
        """
        # split data
        idx1, idx2 = self._split_data(X, node.idx, feature, value)
        # get left revenue posterior rev1 for best price p1
        p1, rev1 = self.teacher._get_best_price(X, idx1, self.prices)
        # get right revenue posterior rev2 for best price p2
        p2, rev2 = self.teacher._get_best_price(X, idx2, self.prices)
        # update best parition node
        self._update_partition(
            node, idx1, idx2, rev1, rev2, p1, p2, feature, value
        )

    def _get_feature_splits(self, X, node, j):
        """
        Sort feature values and return midpoints.

        parameters:
        j (int): positional index of feature in X

        return: List of sorted features
        """
        # get unique features values and sort
        x = np.sort(np.unique(X[node.idx, j])) 
        # return value midpoits
        x = (x[1:] + x[:-1]) / 2
        # trim cuts near the edges
        return x[self.min_samples: -self.min_samples]

    def _remove_children(self, node):
        """
        Remove children from node.
        """
        node.left = None
        node.right = None
        node.feature = None
        node.value = None

    def _split_node(self, X, node):
        """
        Iterate feature values and update optimal split.
        """
        # iterate through features
        for j in range(X.shape[1]):
            # get all split values
            values = self._get_feature_splits(X, node, j)
            # iterate values
            for value in values:
                # update best split
                self._update_best_split(X, node, value, j)

    def _revenue_increased(self, node):
        """
        Check that split occured
        """
        return not (
            (node.left.revenue is None) | (node.right.revenue is None)
        )

    def _split(self, X, node):
        """
        Split node into left and right.
        """
        # add left and right
        self._add_children(node)
        # split node into left and right
        self._split_node(X, node)
        # if VaR is true recursively split
        if self._is_risk_averse(node):
            # assert split params were assigned
            assert (node.feature is not None) & (node.value is not None)
            # split left
            self._split(X, node.left)
            # split right
            self._split(X, node.right)
        else:
            # otherwise, remove left and right
            self._remove_children(node)
            assert (node.feature is None) & (node.feature is None)

    def _add_children(self, node):
        """
        Create node with left and right children.
        """
        node.left = Node()
        node.right = Node()

        return node

    # def _get_best_price(self, X):  # move to teacher model
    #     """
    #     Find best price for max rev.
    #     """
    #     # get revenues
    #     revs = [self.teacher.predict_proba(X, price) * price for price in self.prices]
    #     # get idx of best price
    #     idx = np.argmax([self._total_rev(rev) for rev in revs])
    #     # return price, revenue
    #     return self.prices[idx], revs[idx]

    def _update_attrs(self, node, idx, price, revenue):
        """
        Update node attributes.
        """
        # update attributes
        # node.X = X
        node.idx = idx
        node.price = price
        node.revenue = revenue

    def _update_params(self, node, feature, value):
        """
        Update split parameters.
        """
        node.feature = feature
        node.value = value

    def fit(self, X):
        """
        Fit the Student Prescription Tree.
        """
        # safely convert data to numpy
        X = np.asarray(X)
        # create root node
        self.root_node = Node()
        # create index
        idx = np.arange(len(X))
        # get best price for partition
        price, revenue = self.teacher._get_best_price(X, idx, self.prices)
        # update attribtues
        self._update_attrs(self.root_node, idx, price, revenue)
        # split node
        self._split(X, self.root_node)

    def _transform(self, X, idx, node):
        """
        Recursive function for self.transform()
        """
        # update node test data
        node.X_test = X
        node.idx_test = idx
        # get split params
        feature, value = node.feature, node.value
        # if not leaf node
        if not self._is_leaf(node):
            # split data
            idx1, idx2 = self._split_data(X, idx, feature, value)
            # recursively split
            self._transform(X, idx1, node.left)
            self._transform(X, idx2, node.right)
        else:
            # update revenue
            self.revenue += self._total_rev(node.revenue)

    def transform(self, X):
        """
        Parition X into learned decision tree.
        """
        # get root
        node = self.root_node
        # safely convert data to numpy
        X = np.asarray(X)
        # create index
        idx = np.arange(len(X))
        # transform recursivles
        self._transform(X, idx, node)

    def predict(self, X):
        """
        Predict revenue. Change to prescibe prices.
        """
        # reset revenue
        self.revenue = 0
        # parititon data
        self.transform(X)
        # return predicted revenue
        return self.revenue, self.revenue / len(X)


    def _is_leaf(self, node):
        """
        Check if node is a leaf node.
        """
        # check for split parameters
        return (node.left is None) | (node.right is None)

    def _get_leaf_nodes(self, node=None):
        """
        Collect all leaf nodes into a list
        """
        if node is None:
            node = self.root_node

        if not self._is_leaf(node):
            # recurse left
            self._get_leaf_nodes(node.left)
            # recurse right
            self._get_leaf_nodes(node.right)
        else:
            self.leaf_nodes.append(node)
    
    def prescribe(self, X):
        """
        Use trained policy to prescribe prices based on covariates.
        """
        # init list
        self.leaf_nodes = []
        # fit data to tree 
        self.transform(X)
        # get leave nodes
        self._get_leaf_nodes()
        # get indices
        results = []
        for node in self.leaf_nodes:
            for idx in node.idx_test:
                results += [{"idx": idx, "price": node.price}]
        # covert to dataframe and sort index
        results = pd.DataFrame(results).set_index('idx')\
            .sort_index(ascending=True)
        # return prescribed prices
        return results.price.to_numpy()

    def _get_true_revenue(self, X, optimal_price):
        """
        Given optimal price get true revenue under prescription.
        """
        # total revenue
        rev = ((self.prescribe(X) <= optimal_price) * optimal_price).sum()
        return rev, rev / len(X)
