import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.teachers.BinomGLM import BinomGLM
from src.teachers.OptimalPrescription import OptimalPrescription
from src.DataGenerator import DataGenerator
from src.RiskAverseSPT import RiskAverseSPT 


class ExperimentManager:

    def __init__(self):
        self.seed=121
        self.results = []

    def store_results(self, model, pred_avg_rev, true_avg_rev):
        """
        Store experimental result.
        """
        self.results += [{
            "model": model,
            "dataset": self.dataset,
            "n_samples": self.n_samples,
            "risk_proba": self.risk_proba,
            "pred_avg_rev": pred_avg_rev,
            "true_avg_rev": true_avg_rev,
        }]

    def run_glm(self, X_train, X_test, opt_price_test):
        """
        Train SPT with binomial GLM teacher.
        """
        # init
        glm = BinomGLM()
        # load model
        glm.load_params(
            f'../data/model=binomglm&dataset={self.dataset}&n_samples={self.n_samples}'
        )
        # fit VaR-SPT
        rat = RiskAverseSPT(
            teacher=glm, 
            prices=self.prices, 
            min_samples=10,
            risk_proba=self.risk_proba
        )
        # fit
        rat.fit(X_train)
        # get total and avg rev
        _, pred_avg_rev = rat.predict(X_test)
        # get true revenue 
        _, true_avg_rev = rat._get_true_revenue(X_test, opt_price_test)
        # store results
        self.store_results("binom_glm", pred_avg_rev, true_avg_rev)

    def run_opt(self, X_train, X_test):
        """
        Train SPIT with optimal prescription teacher.
        """
        # init
        opt = OptimalPrescription()

        # fit
        opt.fit(self.optimal_price)

        # fit VaR-SPT
        rat = RiskAverseSPT(
            teacher=opt, 
            prices=self.prices, 
            min_samples=10,
            risk_proba=self.risk_proba
        )
        # fit
        rat.fit(X_train)
        # predict
        _, avg_rev = rat.predict(X_test)
        # return total and avg rev
        self.store_results("opt_price", avg_rev, None)


    def run_experiment(self, dataset, n_samples, risk_proba):
        """
        parameters:
        model (str): 
            one of (binomglm, bart)
        dataset (int): 
            integer identifier of synthetic dataset.
        n_samples (int):
            size of the generated dataset
        """
        # init
        gen = DataGenerator()
        # get dataset
        X, y, price, optimal_price = gen.generate_data(dataset, n_samples, self.seed)
        # get prices
        prices = np.linspace(price.quantile(0.05), price.quantile(0.95), 9)
        # store params
        self.store_params(dataset, n_samples, prices, optimal_price, risk_proba)
        # split data into train and test
        X_train, X_test, opt_price_train, opt_price_test = train_test_split(
            X, optimal_price, test_size=0.33, random_state=52)
        # use teacher binomial GLM
        self.run_glm(X_train, X_test, opt_price_test)
        # use teacher optimal prescription
        self.run_opt(X_train, X_test)

    def store_params(self, dataset, n_samples, prices, optimal_price, risk_proba):
        """
        Store experiment parameters.
        """
        # store params
        self.dataset = dataset
        self.n_samples = n_samples
        self.prices = prices
        self.optimal_price = optimal_price
        self.risk_proba = risk_proba

    def get_results(self):
        """
        Return results as pandas dataframe.
        """
        return pd.DataFrame(self.results)