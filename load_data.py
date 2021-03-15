import numpy as np
from plot import Plot
from tensorflow.keras.utils import to_categorical
from scipy.signal import savgol_filter

class Data:
    def __init__(self, data_path="project"):
        self.data_path = data_path + '/'

    def load(self):

        # load data
        X_test = np.load(self.data_path + "X_test.npy")
        y_test = np.load(self.data_path + "y_test.npy")
        person_train_valid = np.load(self.data_path + "person_train_valid.npy")
        X_train_valid = np.load(self.data_path + "X_train_valid.npy")
        y_train_valid = np.load(self.data_path + "y_train_valid.npy")
        person_test = np.load(self.data_path + "person_test.npy")

        X_gan = np.load(self.data_path + "fake_X_train_valid.npy")
        y_gan = np.load(self.data_path + "fake_y_train_valid.npy")

        # adjust label values
        y_train_valid -= 769
        y_test -= 769
        
        self.dataset = [X_train_valid, y_train_valid]
        self.testset = [X_test, y_test, person_train_valid, person_test]
        self.gan = [X_gan, y_gan]

    def data_splitting(self):

        # # split into train and validation set. Assumes iid
        X_train_valid = self.dataset[0]
        y_train_valid = self.dataset[1]

        ind_valid = np.random.choice(8460, 1500, replace=False)
        ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))
        (Xtrain, Xval) = X_train_valid[ind_train], X_train_valid[ind_valid] 
        (ytrain, yval) = y_train_valid[ind_train], y_train_valid[ind_valid]

        ytrain = to_categorical(ytrain, 4)
        yval = to_categorical(yval, 4)

        self.dataset = [Xtrain, ytrain, Xval, yval]
        self.testset[1] = to_categorical(self.testset[1], 4)
    
    

    def data_filtering(self, D, n=5):
        X_filter = savgol_filter(D, 11, 3)
        return X_filter

    def data_addNoise(self, D, mean=0, std=0.5):
        X_gaussian = D + np.random.normal(loc=mean, scale=std, size=D.shape)
        return X_gaussian

    def data_Average(self, D, stride=2):
        X_mean = np.mean(D.reshape(D.shape[0], D.shape[1], -1, stride), axis=3)
        return X_mean

    def data_maxPooling(self, D, stride=2):
        X_max = np.max(D.reshape(D.shape[0], D.shape[1], -1, stride), axis=3)
        return X_max

    def data_subsampling(self, D, n=2):
        for i in range(n):
            if i == 0:
                X_subsample = D[:, :, i::n]
            else:
                X_subsample = np.vstack((X_subsample, D[:, :, i::n]))
                
        return X_subsample + self.data_addNoise(X_subsample)

    def data_crop(self, D, n=500):
        return D[:, :, 0:n]

    
    def data_augment(self):
        functions_1 = [self.data_crop]
        functions_2 = [self.data_Average, self.data_maxPooling, self.data_subsampling]

        data_aug = []
        for f in functions_1:
            aug_1 = f(self.dataset[0])
            aug_2 = f(self.testset[0])
            data_aug.append([aug_1, aug_2])
        self.data_stack(data_aug)

        data_aug = []
        for f in functions_2:
            aug_1 = f(self.dataset[0])
            aug_2 = f(self.testset[0])
            data_aug.append([aug_1, aug_2])
        self.data_stack(data_aug)
        

    def data_stack(self, data_aug):
        for i in range(len(data_aug)):
            if self.dataset[0].shape != data_aug[i][0].shape:
                self.dataset[0] = data_aug[i][0]
                self.testset[0] = data_aug[i][1]

            else:
                self.dataset[0] = np.vstack((self.dataset[0], data_aug[i][0]))
                self.dataset[1] = np.hstack((self.dataset[1], self.dataset[1]))

                self.testset[0] = np.vstack((self.testset[0], data_aug[i][1]))
                self.testset[1] = np.hstack((self.testset[1], self.testset[1]))
                

    def get_dataset(self):
        return self.dataset

    def get_testset(self):
        return self.testset

    def get_gan(self):
        return self.gan

    def plot_dataset(self):
        figure = Plot()
        figure.show_X(self.dataset[0])

    def print(self):
        print ('Training data shape: {}'.format(self.dataset[0].shape))
        print ('Valid data shape: {}'.format(self.dataset[2].shape))
        print ('Test data shape: {}'.format(self.testset[0].shape))

        print ('Training target shape: {}'.format(self.dataset[1].shape))
        print ('Valid target shape: {}'.format(self.dataset[3].shape))
        print ('Test target shape: {}'.format(self.testset[1].shape))

        print ('Person train/valid shape: {}'.format(self.testset[2].shape))
        print ('Person test shape: {}'.format(self.testset[3].shape))

if __name__ == "__main__":
    data = Data()
    data.load()
    data.data_augment()
    #data.data_splitting()
    #data.print()
    dataset = data.get_dataset()
    testset = data.get_testset()
    gan = data.get_gan()

    #D = dataset[0][:,:,0:500]

    data1 = data.data_filtering(dataset[0])
    data2 = data.data_maxPooling(dataset[0])
    data3 = dataset[0][:, :, 0::2]
    data4 = dataset[0][:, :, 1::2]
    
    figure = Plot()
    # figure.show_y(dataset[0], dataset[1])
    # figure.show_X(dataset[0])

    figure.show_dif([dataset[0], gan[0]], ['Original', 'GAN'])