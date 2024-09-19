import numpy as np


class OptimalPrescription:
    """
    Prescribe the optimal policy based on ground truth data.
    """

    def __init__(self):
        pass

    def fit(self, optimal_price):
        """
        Store data and optimal prices.
        """
        self.optimal_price = np.asarray(optimal_price)

    def _total_rev(self, revenue):
        """
        Total expected revenue in partition.
        """
        return revenue.sum(axis=0).mean()

    def _get_best_price(self, X, idx, prices):
        """
        Find best price for max rev.
        """
        # optimal prices
        opt_prices = self.optimal_price[idx]
        # get best price
        revs = [sum(price <= opt_prices) * price for price in prices]
        # get idx of best price
        idx = np.argmax([self._total_rev(rev) for rev in revs])
        # return price, revenue
        return prices[idx], np.array([[revs[idx]]])