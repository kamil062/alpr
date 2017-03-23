import numpy as np
import math

class Perceptron:

    def __init__(self, n, m, x, d, eta, target):
        self.n = n
        self.m = m
        self.x = x
        self.d = d
        self.ni = eta
        self.beta = -0.0000001
        self.k = 0
        self.Q = 0
        self.target = target
        self.epoch = 0
        self.w = self.initialize_weights
        self.sample = None

    @property
    def initialize_weights(self):
        return np.random.rand(self.m, self.n + 1)

    @property
    def get_sample(self):
        return np.hstack(tuple([1, self.x[self.k]]))

    def calculate_activation_function(self, w, x):
        return 1.0/(1.0 + math.e ** (self.beta * np.dot(w, x.astype('float64'))))

    def adapt_weights(self, w, d, y, x):
        delta = (d - y) * y * (1.0 - y)
        return w + (self.ni * delta * x)

    def calculate_neuron(self, i):
        y = self.calculate_activation_function(self.w[i], self.sample)
        self.w[i] = self.adapt_weights(self.w[i], self.d[self.k, i], y, self.sample)
        self.Q += (self.d[self.k, i] - y) ** 2

    def learn(self, queue):
        try:
            samples = self.x.shape[0]

            if self.x.shape[1] == self.n:
                if self.d.shape[1] == self.m:
                    while True:
                        self.sample = self.get_sample

                        for i in range(0, self.w.shape[0]):
                            self.calculate_neuron(i)

                        self.k += 1

                        if self.k >= samples:
                            self.Q *= 0.5
                            queue.put(self.target / self.Q * 100)
                            self.k = 0
                            self.epoch += 1

                            if self.Q < self.target:
                                return self.w
                else:
                    print "Nieprawidlowa ilosc wyjsc lub nieprawidlowe dane wzorcowe"
            else:
                print "Nieprawidlowa ilosc wejsc lub nieprawidlowe dane uczace"
        except IndexError:
            print "Nieprawidlowa ilosc wejsc lub nieprawidlowe dane uczace"

    def test(self, w, x):
        self.w = w
        self.x = x
        self.k = 0

        try:
            samples = x.shape[0]

            labels = "0123456789abcdefghijklmnoprstuvwxyz"
            platenumber = ""

            while True:
                self.sample = self.get_sample

                answer = []

                for i in range(0, w.shape[0]):
                    y = self.calculate_activation_function(self.w[i], self.sample)
                    answer.append(float("%.2f" % y))

                try:
                    platenumber += labels[answer.index(max(answer))]
                except ValueError:
                    return None

                self.k += 1

                if self.k >= samples:
                    break
            return platenumber.upper()
        except IndexError:
            return None
