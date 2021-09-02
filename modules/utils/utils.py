from sklearn.model_selection import train_test_split
from sklearn import metrics
from .constants import DATASET_MAPPINGS
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


def split_data_set_into_train_and_test(X, y):
    # 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test


def get_accuracy(test, pred):
    return metrics.accuracy_score(test, pred)


def get_precision(test, pred):
    return metrics.precision_score(test, pred, average='micro')


def get_recall(test, pred):
    return metrics.recall_score(test, pred, average='micro')


def get_confusion_matrix(test, pred):
    return metrics.confusion_matrix(test, pred)


def get_r2_score(test, pred):
    return metrics.r2_score(test, pred)


def get_classification_report(test, pred):
    return metrics.classification_report(test, pred)


def get_explained_variance(test, pred):
    return metrics.explained_variance_score(test, pred)


def plot_classification_report():
    pass


def process_dataset(dateset_id, prefix=False):
    dataset = DATASET_MAPPINGS[dateset_id]
    dataset_location, dataset_features, dataset_labels = dataset["location"], dataset["features"], dataset["labels"]
    if prefix:
        dataset_location = "../../../../" + dataset_location
    features = pd.read_csv(dataset_location, usecols=dataset_features).fillna(0)
    labels = pd.read_csv(dataset_location, usecols=dataset_labels).fillna(0)

    return features, labels


def save_model(model, file_name, prefix=False):
    if prefix:
        final_location = "../../../../" + "saved_models/" + file_name
    else:
        final_location = "saved_models/" + file_name

    pickle.dump(model, open(final_location, 'wb'))
    print("Model saved!!")


def load_model(path):
    return pickle.load(open(path, 'rb'))


def plot_scatter_graph(test_x, test_y, pred):
    plt.scatter(test_x, test_y, color='b')
    plt.plot(test_x, pred, color='k')
    plt.show()


def plot_confusion_matrix(matrix):
    plt.matshow(matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def print_stats(labels_test, labels_pred, algo_type, feature_test):
    print("\nSome Statistics: ")
    if algo_type == "c":
        confusion_matrix = get_confusion_matrix(labels_test, labels_pred)
        classification_report = get_classification_report(labels_test, labels_pred)
        print(f"\nAccuracy: {get_accuracy(labels_test, labels_pred)}")
        print(f"\nPrecision: {get_precision(labels_test, labels_pred)}")
        print(f"\nRecall Score: {get_recall(labels_test, labels_pred)}")
        print(f"\nConfusion Matrix: \n{confusion_matrix}")
        print(f"\nClassification Report: \n{classification_report}")
        # plot_graph(feature_test, labels_test, labels_pred)
        plot_confusion_matrix(confusion_matrix)
    if algo_type == "p":
        print(f"\nR2 Score: {get_r2_score(labels_test, labels_pred)}")
        print(f"\nExplained Variance: {get_explained_variance(labels_test, labels_pred)}")
        # plot_scatter_graph(feature_test, labels_test, labels_pred)


def display_graphs():
    pass
