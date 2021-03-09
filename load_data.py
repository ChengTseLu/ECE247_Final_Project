import numpy as np


class Data:
    def __init__(self, data_path="project"):
        self.data_path = data_path + '/'

    def setup(self, Print=False):

        # load data
        X_test = np.load(self.data_path + "X_test.npy")
        y_test = np.load(self.data_path + "y_test.npy")
        person_train_valid = np.load(self.data_path + "person_train_valid.npy")
        X_train_valid = np.load(self.data_path + "X_train_valid.npy")
        y_train_valid = np.load(self.data_path + "y_train_valid.npy")
        person_test = np.load(self.data_path + "person_test.npy")

        # adjust label values
        y_train_valid -= 769
        y_test -= 769

        if Print:
            print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
            print ('Test data shape: {}'.format(X_test.shape))
            print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
            print ('Test target shape: {}'.format(y_test.shape))
            print ('Person train/valid shape: {}'.format(person_train_valid.shape))
            print ('Person test shape: {}'.format(person_test.shape))

        # split into train and validation set. Assumes iid
        perm = np.random.permutation(X_train_valid.shape[0])
        numTrain = int(0.8*X_train_valid.shape[0])
        numVal = X_train_valid.shape[0] - numTrain
        Xtrain = X_train_valid[perm[0:numTrain]]
        ytrain = y_train_valid[perm[0:numTrain]]
        Xval = X_train_valid[perm[numTrain: ]]
        yval = y_train_valid[perm[numTrain: ]]

        self.dataset = [Xtrain, ytrain, Xval, yval]
        self.testset = [X_test, y_test, person_train_valid, person_test]

    def data_filtering(self):
        pass

    def data_augment(self):
        pass

    def data_addNoise(self, mean=0, std=1):
        self.dataset[0] += np.random.normal(loc=mean, scale=std, size=self.dataset[0].shape)

    def get_dataset(self):
        return self.dataset

    def get_testset(self):
        return self.testset

if __name__ == "__main__":
    data = Data()
    data.setup(True)
    data.data_addNoise()
    