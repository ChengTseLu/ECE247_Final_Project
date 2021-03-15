from load_data import Data
from model import Model
from train import Train


if __name__ == "__main__":

    data = Data()
    data.load()
    data.data_augment()
    data.data_splitting()
    data.print()
    dataset = data.get_dataset()
    testset = data.get_testset()
    
    models = Model(dataset[0].shape[1:], 50)
    m = models.ResNet()
    m.summary()

    # train = Train(m, dataset, testset, 50, 32, 'adam', 'sparse_categorical_crossentropy')
    train = Train(m, dataset, testset, 100, 200, 'adam', 'categorical_crossentropy')
    train.training()
    train.evaluate()

    


# tensorboard --logdir logs/scalars --port=7000