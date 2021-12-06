import matplotlib.pylab as plt
import numpy as np
import gzip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import cv2

class Kohonen:
    # initialization
    def __init__(self, m, n, model=None, net_dim=(30, 30), learning_rate=0.5, iterations=100, n_classes=10):
        if model is None:
            model = np.random.random((net_dim[0], net_dim[1], m))*0.01
        self.model = model
        self.net_dim = net_dim
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.init_radius = np.average(np.array(net_dim)) / 2
        self.m = m
        self.n = n
        self.n_classes = n_classes
        self.time_constant = iterations / np.log(self.init_radius)
        self.labels = np.zeros(
            (self.model.shape[0], self.model.shape[1], self.n_classes))

    # competition
    def fittest_neuron(self, feature):
        """
        Finds fittest neuron w.r.t. given input features x and model

        :return: the neuron and its index in model.
        """
        # calculate the distance between each neuron and the input using vectorized operation
        distance = np.sqrt(np.sum((self.model - feature) ** 2, axis=2))
        pos = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
        return (self.model[pos], pos)

    # Cooperation
    def decay_radius(self, i):
        return self.init_radius * np.exp(-i / self.time_constant)

    def decay_learning_rate(self, i):
        return self.learning_rate * np.exp(-i / self.iterations)

    # gaussian neighboring
    def h(self, distance, radius):
        return np.exp(-distance / (2 * (radius**2)))

    # Adaption
    def train(self, x_train):
        zx = np.arange(0, self.net_dim[0], 1)
        zy = np.arange(0, self.net_dim[1], 1)
        zx, zy = np.meshgrid(zx, zy, indexing='ij')
        mesh_init = np.array([zx, zy])

        for i in range(self.iterations):
            # sample = x_train[np.random.randint(0, n, 1)]
            for feature in x_train:

                # find the fittest
                fittest, fittest_idx = self.fittest_neuron(feature)

                # decay the SOM parameters
                r = self.decay_radius(i)
                l = self.decay_learning_rate(i)

                # update weight vector
                mesh_init[0] = mesh_init[0] - fittest_idx[0]
                mesh_init[1] = mesh_init[1] - fittest_idx[1]
                mesh = np.sqrt(np.sum(mesh_init ** 2, axis=0))
                neighbor_mask = mesh < r
                if len(neighbor_mask.flatten()) > 0:
                    mesh[neighbor_mask] = self.h(mesh[neighbor_mask], r)
                    self.model[neighbor_mask] = self.model[neighbor_mask] + l * np.multiply(
                        mesh[neighbor_mask][:, np.newaxis], (feature - self.model)[neighbor_mask])

    def error(self, x):
        distances = np.empty((x.shape[0],))
        for idx, feature in enumerate(x):
            fittest, fittest_idx = self.fittest_neuron(feature)
            distances[idx] = ((fittest - feature) ** 2).sum()
        return distances.mean()

    def label_neurons(self, x_train, y_train):
        # majority vote on reiterating train data and labeling winner node
        for idx, feature in enumerate(x_train):
            distance = np.sqrt(np.sum((self.model - feature) ** 2, axis=2))
            pos = np.unravel_index(
                np.argmin(distance, axis=None), distance.shape)
            self.labels[pos[0], pos[1], y_train[idx]] += 1
        self.labels = np.argmax(self.labels, axis=2)
        return self.labels

    def accuracy(self, x_test, y_test):
        t = 0
        for idx, feature in enumerate(x_test):
            distance = np.sqrt(np.sum((self.model - feature) ** 2, axis=2))
            pos = np.unravel_index(
                np.argmin(distance, axis=None), distance.shape)
            if self.labels[pos[0], pos[1]] == y_test[idx]:
                t += 1
        return t / len(y_test)

    def visualize_map(self, x_test, y_test):
        wmap = {}
        im = 0
        for x, t in zip(x_test[:1000], y_test[:1000]):
            distance = np.sqrt(np.sum((self.model - x) ** 2, axis=2))
            pos = np.unravel_index(
                np.argmin(distance, axis=None), distance.shape)
            wmap[pos] = im
            plt.text(pos[0],  pos[1],  str(t), color=plt.cm.rainbow(
                t / 10.), fontdict={'weight': 'bold',  'size': 11})
            im = im + 1
        plt.axis([0, self.model.shape[0], 0,  self.model.shape[1]])
        plt.show()
