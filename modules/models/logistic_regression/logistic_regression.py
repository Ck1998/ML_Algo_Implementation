from modules.utils.utils import split_data_set_into_train_and_test, get_r2_score, process_dataset, save_model
from sklearn.linear_model import LogisticRegression
from modules.utils.constants import DATASET_MAPPINGS


class LogisticRegressionTest:

    def __init__(self, dataset_id: int, prefix=False):
        self.dataset_id = dataset_id
        self.prefix = prefix
        self.features, self.labels = process_dataset(dateset_id=dataset_id, prefix=self.prefix)
        self.features_train, self.features_test, self.labels_train, self.labels_test = \
            split_data_set_into_train_and_test(self.features, self.labels)
        self.clf = LogisticRegression()

    def save(self, suffix):
        save_model(model=self.clf,
                   file_name=f"logistic_regression_{DATASET_MAPPINGS[self.dataset_id]['name']}_{suffix}.model",
                   prefix=self.prefix)

    def run(self):
        self.clf.fit(self.features_train, self.labels_train)
        labels_predict = self.clf.predict(self.features_test)
        print(f"R2 Score: {get_r2_score(self.labels_test, labels_predict)}")
        if input("You want to save the model ? [Y/N]: ").lower() == "y":
            self.save(input("Enter file suffix: "))


if __name__ == "__main__":
    LogisticRegressionTest(dataset_id=2, prefix=True).run()
