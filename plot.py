import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, loss_hist=None, dataset=None):
        if loss_hist != None:
            self.hist = loss_hist.history
        self.dataset = dataset

    def draw(self):
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

    def show_dataset(self):
        plt.figure()

        avg = np.mean(self.dataset[0], axis=0)
        name = []
        i = 0
        for channel in avg:
            plt.plot(channel)
            i += 1
            name.append('channel: ' + str(i))
            
        plt.legend(name)
        plt.show()

        