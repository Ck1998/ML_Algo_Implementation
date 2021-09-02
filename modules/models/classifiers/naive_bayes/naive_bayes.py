from modules.utils.utils import split_data_set_into_train_and_test, print_stats, process_dataset, save_model
from sklearn.naive_bayes import GaussianNB
from modules.utils.constants import DATASET_MAPPINGS


class NaiveBayesTest:

    def __init__(self, dataset_id: int, prefix=False):
        self.dataset_id = dataset_id
        self.prefix = prefix
        self.features, self.labels = process_dataset(dateset_id=dataset_id, prefix=self.prefix)
        self.features_train, self.features_test, self.labels_train, self.labels_test = \
            split_data_set_into_train_and_test(self.features, self.labels)
        self.labels_train = self.labels_train.values.ravel()
        self.clf = GaussianNB()

    def save(self, suffix):
        save_model(model=self.clf,
                   file_name=f"stochastic_gradient_descent_classifier_"
                             f"{DATASET_MAPPINGS[self.dataset_id]['name']}_{suffix}.model",
                   prefix=self.prefix)

    def run(self):
        self.clf.fit(self.features_train, self.labels_train)
        labels_predict = self.clf.predict(self.features_test)
        print_stats(labels_test=self.labels_test, labels_pred=labels_predict, algo_type="c",
                    feature_test=self.features_test)
        if input("\nYou want to save the model ? [Y/N]: ").lower() == "y":
            self.save(input("Enter file suffix: "))


if __name__ == "__main__":
    NaiveBayesTest(dataset_id=2, prefix=True).run()
