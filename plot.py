import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, loss_hist=None):
        if loss_hist != None:
            self.hist = loss_hist.history

    def show_loss(self):
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(self.hist['loss'])
        plt.plot(self.hist['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])

        plt.subplot(1, 2, 2)
        plt.plot(self.hist['acc'])
        plt.plot(self.hist['val_acc'])
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])

        plt.show()

    # plot data in terms of channels
    def show_X(self, X):

        avg = np.mean(X, axis=0)
        name = []
        i = 0
        for channel in avg:
            plt.plot(channel)
            i += 1
            name.append('channel: ' + str(i))
            
        plt.legend(name)
        plt.show()

    # plot data in terms of classes
    def show_y(self, X, y):
        plt_lst = [221, 222, 223, 224]
        name = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']

        y_ = np.argmax(y, axis=1)
        for i in range(4):
            X_sub = np.mean(X[y_==i], axis=(0,1))

            plt.subplot(plt_lst[i])
            plt.plot(X_sub)
            plt.title(name[i])


        plt.show()

    def show_dif(self, X, name):
        for d in X:
            print(d.shape)
            avg_d = np.mean(d, axis=(0,1))
            plt.plot(avg_d)

        plt.legend(name)
        plt.show()