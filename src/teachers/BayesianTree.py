import pymc as pm
import numpy as np
import arviz as az
from scipy import special
import json 
import pandas as pd
import pymc_bart as pmb


class BayesianTree:
    """
    Bayesian Additive Regression Tree (BART).
    """

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
        X = pd.concat([price, X], axis=1)

        import pymc as pm
        import pymc_bart as pmb
        with pm.Model() as self.model:
            X_ = pm.MutableData("X", X)

            # Define the BART model
            logits = pmb.BART('logits', X_, y)
            
            # Convert logits to probabilities
            p = pm.Deterministic('p', pm.math.sigmoid(logits))
            
            # Likelihood
            y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
            
            # Inference
            self.trace = pm.sample(draws=draws, tune=1000, cores=2)

        # store params
        self.probas = np.concatenate(self.trace.posterior.p)

    def predict_proba(self, price:int, X):
        """
        Given a prive and covariates predict purchase probability.

        parameters:
        price (int): price of product.
        prices (array): historical prices.
        X (array): covariates 

        return:
        posterior purchase probability for each row in X. 
        """
        price = pd.Series(np.repeat(price, len(X)))
        X = pd.concat([price, X], axis=1)

        # Use the fitted BART model to predict on new data
        with self.model:
            pm.set_data({"X": X})
            # Use the posterior predictive to generate predictions
            ppc = pm.sample_posterior_predictive(self.trace, random_seed=42, var_names=['p'])

        # get predictions
        pred_probas = np.concatenate(ppc.posterior_predictive['p'], axis=0)
        return pred_probas.T

    def _get_best_price(self, X, idx, prices):
        """
        Prescribe the price which maximizes revenue.
        """
        # get revenues
        revs = [self.predict_proba(X[idx], price) * price for price in prices]
        # get idx of best price
        idx = np.argmax([self._total_rev(rev) for rev in revs])
        # return price, revenue
        return prices[idx], revs[idx]

    def _total_rev(self, revenue):
        """
        Total expected revenue in partition.
        """
        return revenue.sum(axis=0).mean()

    def save_params(self, PATH):
        """
        Save trained model params to local file system.
        """
        params = {
            "probas": np.concatenate(self.trace['posterior']['alpha'], axis=0).tolist()
        }

        with open(PATH, 'w') as f:
            json.dump(params, f)

    def load_params(self, PATH):
        """
        Load model params from local file system.
        """
        with open(PATH, 'r') as f:
            params = json.load(f)

        self.alpha = np.array(params['alpha'])
        self.beta = np.array(params['beta'])
