from modules.models.classifiers.random_forest.random_forest import RandomForestTest
from modules.models.predictors.linear_regression.linear_regression import LinearRegressionTest
from .logistic_regression.logistic_regression import LogisticRegressionTest


ALGO_MAPPING = {
    1: {
        "name": "Classifiers",
        "datasets": [1, 2, 3, 4],
        "algorithms": {
            1: {
                "name": "Random_Forest_Classifier",
                "obj": RandomForestTest
            },
            2: {
                "name": "Logistic_Regression",
                "obj": LogisticRegressionTest
            }
        }
    },
    2: {
        "name": "Predictors",
        "datasets": [],
        "algorithms": {
            1: {
                "name": "Linear_Regression",
                "obj": LinearRegressionTest
            }
        }
    }
}
