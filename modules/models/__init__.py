from modules.models.classifiers.random_forest.random_forest import RandomForestTest
from modules.models.classifiers.svm.svm import SVMTest
from modules.models.classifiers.knn.knn import KNNTest
from modules.models.classifiers.sgd.sgd import StochasticGradientDescentTest
from modules.models.classifiers.naive_bayes.naive_bayes import NaiveBayesTest
from modules.models.predictors.linear_regression.linear_regression import LinearRegressionTest
from modules.models.predictors.lasso_regression.lasso_regression import LassoRegressionTest
from modules.models.predictors.ridge_regression.ridge_regression import RidgeRegressionTest
from modules.models.predictors.gamma_regressor.gamma_regressor import GammaRegressorTest
from modules.models.classifiers.logistic_regression.logistic_regression import LogisticRegressionTest
from modules.models.auxillary.face_detection.face_detection import FaceDetection


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
                "name": "Stochastic_Gradient_Descent",
                "obj": StochasticGradientDescentTest
            },
            3: {
                "name": "K_Nearest_Neighbour",
                "obj": KNNTest
            },
            4: {
                "name": "Support_Vector_Machine",
                "obj": SVMTest
            },
            5: {
                "name": "Naive_Bayes",
                "obj": NaiveBayesTest
            },
            6: {
                "name": "Logistic_Regression",
                "obj": LogisticRegressionTest
            }
        }
    },
    2: {
        "name": "Predictors",
        "datasets": [5, 6, 7, 8],
        "algorithms": {
            1: {
                "name": "Linear_Regression",
                "obj": LinearRegressionTest
            },
            2: {
                "name": "Lasso_Regression",
                "obj": LassoRegressionTest
            },
            3: {
                "name": "Ridge_Regression",
                "obj": RidgeRegressionTest
            },
            4: {
                "name": "Gamma_Regression",
                "obj": GammaRegressorTest
            }
        }
    },
    3: {
        "name": "Auxiliary",
        "datasets": [],
        "algorithms": {
            1: {
                "name": "Face_Detection_Using_OpenCV",
                "obj": FaceDetection
            }
        }
    }
}
