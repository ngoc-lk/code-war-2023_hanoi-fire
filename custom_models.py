#
# AI Wonder Custom Models
#

from flaml import AutoML
from flaml.automl.model import SKLearnEstimator
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC as SVMClassifier, SVR as SVMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Lasso as LassoRegression, Ridge as RidgeRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from flaml import tune


# Custom estimators that wrap various machine learning algorithms

### Decision Trees
decision_tree_space = {
    "max_depth": {
        "domain": tune.randint(lower=1, upper=50),
        "init_value": 10,
    },
    "min_samples_split": {
        "domain": tune.lograndint(lower=2, upper=20),
        "init_value": 2,
    },
    "min_samples_leaf": {
        "domain": tune.lograndint(lower=1, upper=20),
        "init_value": 1,
    },
    "max_leaf_nodes": {
        "domain": tune.randint(lower=2, upper=100),
        "init_value": 50,
    },
    "min_impurity_decrease": {
        "domain": tune.uniform(lower=0.0, upper=0.5),
        "init_value": 0.0,
    },
}

class DecisionTreeClassifierEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = DecisionTreeClassifier

    @classmethod
    def search_space(cls, data_size, task):
        return decision_tree_space

class DecisionTreeRegressorEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = DecisionTreeRegressor

    @classmethod
    def search_space(cls, data_size, task):
        return decision_tree_space

### GradientBoosting
gradient_boosting_space = {
    "n_estimators": {
        "domain": tune.lograndint(lower=10, upper=1000),
        "init_value": 100,
    },
    "learning_rate": {
        "domain": tune.loguniform(lower=0.01, upper=1.0),
        "init_value": 0.1,
    },
    "max_depth": {
        "domain": tune.randint(lower=1, upper=10),
        "init_value": 3,
    },
    "min_samples_split": {
        "domain": tune.lograndint(lower=2, upper=20),
        "init_value": 2,
    },
    "min_samples_leaf": {
        "domain": tune.lograndint(lower=1, upper=20),
        "init_value": 1,
    },
    "subsample": {
        "domain": tune.uniform(lower=0.5, upper=1.0),
        "init_value": 1.0,
    },
    "max_leaf_nodes": {
        "domain": tune.lograndint(lower=4, upper=32),
        "init_value": None,
    },
    "min_impurity_decrease": {
        "domain": tune.uniform(lower=0.0, upper=0.2),
        "init_value": 0.0,
    },
}

class GradientBoostingClassifierEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = GradientBoostingClassifier

    @classmethod
    def search_space(cls, data_size, task):
        return gradient_boosting_space

class GradientBoostingRegressorEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = GradientBoostingRegressor

    @classmethod
    def search_space(cls, data_size, task):
        return gradient_boosting_space

### CatBoost
catboost_space = {
    "iterations": {
        "domain": tune.choice([100, 200, 300]),
        "init_value": 100,
    },
    "learning_rate": {
        "domain": tune.choice([0.01, 0.05, 0.1]),
        "init_value": 0.1,
    },
    "depth": {
        "domain": tune.choice([6, 8, 10]),
        "init_value": 6,
    },
    "l2_leaf_reg": {
        "domain": tune.uniform(lower=1.0, upper=10.0),
        "init_value": 3.0,
    },
}

class CatBoostClassifierEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = CatBoostClassifier

    @classmethod
    def search_space(cls, data_size, task):
        return catboost_space

class CatBoostRegressorEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = CatBoostRegressor

    @classmethod
    def search_space(cls, data_size, task):
        return catboost_space

### KNeighbors
knn_space = {
    "n_neighbors": {
        "domain": tune.lograndint(lower=1, upper=100),
        "init_value": 5,
    },
    "weights": {
        "domain": tune.choice(["uniform", "distance"]),
        "init_value": "uniform",
    },
    "algorithm": {
        "domain": tune.choice(["auto", "ball_tree", "kd_tree", "brute"]),
        "init_value": "auto",
    },
    "leaf_size": {
        "domain": tune.lograndint(lower=10, upper=100),
        "init_value": 30,
    },
    "p": {
        "domain": tune.choice([1, 2]),
        "init_value": 2,
    },
}

class KNeighborsClassifierEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = KNeighborsClassifier

    @classmethod
    def search_space(cls, data_size, task):
        return knn_space

class KNeighborsRegressorEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = KNeighborsRegressor

    @classmethod
    def search_space(cls, data_size, task):
        return knn_space

### Support Vector Machine
svm_space = {
    "C": {
        "domain": tune.loguniform(lower=0.01, upper=100),
        "init_value": 1.0,
    },
    "kernel": {
        "domain": tune.choice(["linear"]),              # "poly", "rbf", "sigmoid",
        "init_value": "linear",
    },
    "degree": {
        "domain": tune.randint(lower=1, upper=5),
        "init_value": 3,
    },
    "gamma": {
        "domain": tune.choice(["scale", "auto"]),
        "init_value": "scale",
    },
    "coef0": {
        "domain": tune.uniform(lower=0.0, upper=10.0),
        "init_value": 0.0,
    },
    "cache_size": {
        "domain": tune.choice([200, 500, 1000, 2000]),
        "init_value": 200,
    },
}

class SVMClassifierEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = SVMClassifier
        self.params.update({"probability": True})
        
    @classmethod
    def search_space(cls, data_size, task):
        return svm_space

class SVMRegressorEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = SVMRegressor

    @classmethod
    def search_space(cls, data_size, task):
        return svm_space

### Multi-layer Perceptron
mlp_space = {
    "hidden_layer_sizes": {
        "domain": tune.lograndint(lower=10, upper=200),
        "init_value": 100,
    },
    "activation": {
        "domain": tune.choice(["identity", "logistic", "tanh", "relu"]),
        "init_value": "relu",
    },
    "solver": {
        "domain": tune.choice(["lbfgs", "sgd", "adam"]),
        "init_value": "adam",
    },
    "alpha": {
        "domain": tune.loguniform(lower=0.0001, upper=0.05),
        "init_value": 0.0001,
    },
    "learning_rate": {
        "domain": tune.choice(["constant", "invscaling", "adaptive"]),
        "init_value": "constant",
    },
}

class MLPClassifierEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = MLPClassifier

    @classmethod
    def search_space(cls, data_size, task):
        return mlp_space

class MLPRegressorEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = MLPRegressor

    @classmethod
    def search_space(cls, data_size, task):
        return mlp_space

# Linear Regression (Lasso and Ridge Regression)
linear_regression_space = {
    "alpha": {
        "domain": tune.loguniform(lower=0.0001, upper=10),
        "init_value": 1.0,
    },
    "solver": {
        "domain": tune.choice(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
        "init_value": 'auto',
    },
    "max_iter": {
        "domain": tune.randint(lower=100, upper=1000),
        "init_value": 1000,
    },
    "tol": {
        "domain": tune.loguniform(lower=1e-6, upper=1e-3),
        "init_value": 1e-4,
    }
}

class LassoRegressionEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = LassoRegression

    @classmethod
    def search_space(cls, data_size, task):
        return {k: v for k, v in linear_regression_space.items() if k != "solver"}

class RidgeRegressionEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = RidgeRegression

    @classmethod
    def search_space(cls, data_size, task):
        return linear_regression_space

# Multinomial Naive Bayes
multinomial_nb_space = {
    "alpha": {
        "domain": tune.loguniform(lower=0.0001, upper=2.0),
        "init_value": 1.0,
    },
    "fit_prior": {
        "domain": tune.choice([True, False]),
        "init_value": True,
    }
}

class MultinomialNBEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = MultinomialNB

    @classmethod
    def search_space(cls, data_size, task):
        return multinomial_nb_space

# Gaussian Naive Bayes
gaussian_nb_space = {
    "var_smoothing": {
        "domain": tune.loguniform(lower=1e-10, upper=1e-8),
        "init_value": 1e-9,
    }
}

class GaussianNBEstimator(SKLearnEstimator):
    def __init__(self, task, n_jobs=-1, **config):
        super().__init__(task, **config)
        self.estimator_class = GaussianNB

    @classmethod
    def search_space(cls, data_size, task):
        return gaussian_nb_space

