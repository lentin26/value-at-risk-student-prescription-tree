import pymc as pm
import numpy as np
import arviz as az
from scipy import special


class BetaBinomial:

    def __init__(self):
        pass

    def plot_trace(self):
        """
        Plot trace
        """
        pm.plot_trace(self.trace)

    def summary(self, round_to=2):
        """
        Print summary of results.
        """
        # Summarize the results
        return az.summary(self.trace, round_to=round_to)

    def fit(self, price, X, y, draws=100):
        """
        Fit Beta Binomial model using PYMC.
        """
        # dim of covariate space
        d = X.shape[1]
        with pm.Model() as model:
            # prior scale
            theta_a = pm.InverseGamma("theta_a", 1, 1)
            theta_b = pm.InverseGamma("theta_b", 1, 1)

            # slopes
            alpha = pm.MvNormal("alpha", mu=[0] * d, cov=theta_a*np.identity(d))
            beta = pm.MvNormal("beta", mu=[0] * d, cov=theta_b*np.identity(d))

            p_bar = pm.Deterministic(
                "p_bar", 
                pm.math.invlogit(
                    pm.math.dot(X, alpha) + pm.math.dot(X, beta) * price
                )
            )

            # expected success rate
            A = pm.Binomial("y", observed=y, n=[1] * len(y), p=p_bar)
            self.trace = pm.sample(draws=draws, tune=2000, target_accept=0.95, return_inferencedata=True)

    def predict_proba(self, price: int, X):
        """
        Given a prive and covariates predict purchase probability.

        parameters:
        price (int): price of product.

        return:
        posterior purchase probability for each row in X. 
        """
        # get linear model paramters
        alpha = np.concatenate(self.trace['posterior']['alpha'], axis=0)
        beta = np.concatenate(self.trace['posterior']['beta'], axis=0)

        # create price vector
        price = np.repeat(price, X.shape[0]).reshape(-1, 1)

        # get predicted proba posteriors
        pred_proba = special.expit(
            X @ alpha.T + price * (X @ beta.T)
        )

        # safely convert to numpy array
        return np.asarray(pred_proba)
