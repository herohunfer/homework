import numpy as np
from sklearn.utils import shuffle


def sigmoid(x):
    """
    Sigmoid (logistic) function
    :param x: array-like shape(n_sample, n_feature)
    :return: simgoid value (array like)
    """
    # TODO sigmoid function
    return 1. / (1. + np.exp(-x))


def dsigmoid(x):
    """
    Derivative of sigmoid function
    :param x: array-like shape(n_sample, n_feature)
    :return: derivative value (array like)
    """
    # TODO dsigmoid function
    sig = sigmoid(x)
    return sig * (1 - sig)


def tanh(x):
    """
    Tanh function
    :param x: array-like shape(n_sample, n_feature)
    :return: tanh value (array like)
    """
    # TODO tanh function
    return 2. / (1 + np.exp(-2 * x)) - 1.


def dtanh(x):
    """
    Derivative of tanh function
    :param x: array-like shape(n_sample, n_feature)
    :return: derivative value (array like)
    """
    # TODO dtanh function
    return 1 - tanh(x) ** 2


def softmax(X):
    """
    softmax function
    :param X:
    :return:
    """
    # TODO softmax function
    return (np.exp(X).T / np.sum(np.exp(X), axis=1)).T


class MLP:
    def __init__(self, input_size, output_size, hidden_layer_size=[100], batch_size=200, activation="sigmoid",
                 output_layer='softmax', loss='cross_entropy', lr=0.01, reg_lambda=0.0001, momentum=0.9, verbose=10):
        """
        Multilayer perceptron Class
        :param input_size: int, input size (n_feature)
        :param output_size: int,  output node size (n_class)
        :param hidden_layer_size: list of int, each int represents the size of a hidden layer
        :param batch_size: int, batch_size
        :param activation: string, activation function ['sigmoid', 'tanh']
        :param output_layer: string, output layer type ['softmax']
        :param loss: string, loss type ['cross_entropy']
        :param lr: float, learning rate
        :param reg_lambda: float, lambda of regularization
        :param verbose: int, print flag
        :param momentum: float, momentum
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.batch_size = batch_size
        self.n_layers = len(hidden_layer_size)  # only hidden layers
        self.activation = activation
        self.verbose = verbose

        if activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_dfunc = dsigmoid
        elif activation == 'tanh':
            self.activation_func = tanh
            self.activation_dfunc = dtanh
        else:
            raise ValueError("Currently only supoort 'sigmoid' or 'tanh' activations func!")

        self.loss = loss
        if output_layer == 'softmax':
            self.output_layer = softmax
        else:
            raise ValueError('Currently only Support softmax output_layer!')

        self.nclass = output_size

        self.weights = []  # store weights
        self.bias = []  # store bias
        self.layers = []  # store forwarding activation values
        self.deltas = []  # store errors for backprop

    def get_weight_bound(self, fan_in, fan_out):
        """
        Generate bound value for random weights initialization
        :param fan_in: layer input size
        :param fan_out: layer output size
        :return: float, bound
        """
        if self.activation == 'sigmoid':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation == 'tanh':
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        return init_bound

    def fit(self, X, y, max_epochs, shuffle_data):
        """
        fit the model given data X and label y
        :param X: array-like, shape(n_samples, n_features)
        :param y: array-like, shape(n_samples, 1)
        :param max_epochs: int, max iterations
        :param shuffle_data: bool, if shuffle the data.
        :return: MLP model object
        """
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("Shapes of X and y don't fit!")

        # generate weights
        # Weights and bias connecting input layer and first hidden layer
        init_bound = self.get_weight_bound(n_features, self.hidden_layer_size[0])
        self.weights.append(np.random.uniform(-init_bound, init_bound, size=(n_features, self.hidden_layer_size[0])))
        self.bias.append(np.random.uniform(-init_bound, init_bound, self.hidden_layer_size[0]))

        # Weights and bias connecting hidden layers
        for i in range(1, len(self.hidden_layer_size)):
            init_bound = self.get_weight_bound(self.hidden_layer_size[i - 1], self.hidden_layer_size[i])
            self.weights.append(np.random.uniform(-init_bound, init_bound,
                                                  size=(self.hidden_layer_size[i - 1], self.hidden_layer_size[i])))
            self.bias.append(np.random.uniform(-init_bound, init_bound, self.hidden_layer_size[i]))

        # Weights and bias connecting last hidden layer and output layer
        init_bound = self.get_weight_bound(self.hidden_layer_size[-1], self.output_size)
        self.weights.append(
            np.random.uniform(-init_bound, init_bound, size=(self.hidden_layer_size[-1], self.output_size)))
        self.bias.append(np.random.uniform(-init_bound, init_bound, self.output_size))

        # pre-allocate memory for both activations and errors
        # for input layer
        self.layers.append(np.empty((self.batch_size, self.input_size)))
        # for hidden layers
        for i in range(0, self.n_layers):
            self.layers.append(np.empty((self.batch_size, self.hidden_layer_size[i])))
            self.deltas.append(np.empty((self.batch_size, self.hidden_layer_size[i])))
        # for output layer
        self.layers.append(np.empty((self.batch_size, self.output_size)))
        self.deltas.append(np.empty((self.batch_size, self.output_size)))

        # main loop
        for i in xrange(max_epochs):
            # shuffle data
            # TODO shuffle data
            np.random.shuffle(X)
            # iterate every batch
            for batch in xrange(0, n_samples, self.batch_size):
                # TODO call forward function
                self.forward(X[batch:batch+self.batch_size])
                # TODO call backward function
                self.backward(X[batch:batch+self.batch_size], y)
            if i % self.verbose == 0:
                # Compute Loss and Training Accuracy
                loss = self.compute_loss(X, y)
                acc = self.score(X, y)
                print('Epoch {}: loss = {}, accuracy = {}'.format(i, loss, acc))

        return self

    def compute_loss(self, X, y):
        """
        Compute loss
        :param X: data, array-like, shape(n_sample, n_feature)
        :param y: label, array-like, shape(n_sample, 1)
        :return: loss value
        """
        n_samples = X.shape[0]
        probs = self.forward(X)

        # TODO Calculating the loss
        m = y.shape[0]
        b = np.zeros((y.shape[0], self.output_size))
        for i in range(m):
            b[i, y[i]] = 1
        #data_loss = 0.5 * np.sum((b[:n_samples] - probs) ** 2)
        data_loss = -1. * np.sum(b[:n_samples] * np.log(probs))
        # TODO Add regularization term to loss
        data_loss += + self.reg_lambda/2. * np.sum(self.weights[-1] * self.weights[-1])
        return 1. / n_samples * data_loss

    def forward(self, X):
        # input layer
        self.layers[0] = X

        # TODO hidden layers
        for i in range(1, self.n_layers+1):
            self.layers[i] = self.activation_func(np.dot(self.layers[i-1],  self.weights[i-1])+ self.bias[i-1])

        # TODO output layer (Note here the activation is using output_layer func)
        self.layers[-1] = self.output_layer(self.activation_func(np.dot(self.layers[self.n_layers],
                                                   self.weights[self.n_layers])+ self.bias[self.n_layers]))
        return self.layers[-1]

    def backward(self, X, y):
        if self.loss == 'cross_entropy':
            self.deltas[-1] = self.layers[-1]
            # cross_entropy loss backprop
            n_samples = X.shape[0]

            m = y.shape[0]
            b = np.zeros((y.shape[0], self.output_size))
            for i in range(m):
                b[i,y[i]] = 1

            #print(b[:n_samples].shape)
            #print(self.deltas[-1].shape)
            #self.deltas[range(n_samples), b] -= 1
            #self.deltas /= n_samples
            self.deltas[-1] = (self.deltas[-1] - b[:n_samples]) * self.activation_dfunc(self.layers[-1])
            #dW = self.deltas[-1] * self.layers[self.n_layers] + self.reg_lambda * self.weights[-1]
            dW = np.dot(self.layers[self.n_layers].T, self.deltas[-1]) + self.reg_lambda * self.weights[-1]
            dB = np.sum(self.deltas[-1], axis=0)
            self.weights[-1] += -1. * self.lr * dW
            #print(self.weights[-1].shape)
            #print(self.bias[-1].shape)
            #print(self.deltas[-1].shape)
            #print(dB.shape)
            self.bias[-1]+= -1. * self.lr * dB

            for i in range(self.n_layers-1, -1, -1):
                # TODO update deltas
                ai = self.activation_dfunc(self.layers[i+1]) # ignore input layer
                self.deltas[i] = np.dot(self.deltas[i+1], self.weights[i+1].T)
                #print(self.deltas[i].shape)
                #print(ai.shape)
                self.deltas[i] *= ai #* ai
                # TODO update weights
                dW = np.sum(self.deltas[i]*ai) + self.reg_lambda * self.weights[i]
                dB = np.sum(self.deltas[i], axis=0)
                self.weights[i] += -1. * self.lr * dW
                self.bias[i] += -1. * self.lr * dB

    def predict(self, X):
        """
        predicting probability outputs
        :param X: array-like, shape(n_samples, n_features)
        :return: array-like, predicted probabilities
        """
        return self.forward(X)

    def score(self, X, y):
        """
        compute accuracy
        :param X: array-like, shape(n_samples, n_features)
        :param y: ground truth labels array-like, shape(n_samples, 1)
        :return: float, accuracy
        """
        n_samples = X.shape[0]
        # TODO compute accuracy
        probs = self.predict(X)
        preds = np.argmax(probs, axis=1)
        print(probs)
        acc = np.mean(preds == y[:n_samples])
        return acc


def my_mlp():
    # from sklearn.datasets import fetch_mldata
    # mnist = fetch_mldata("MNIST original")
    # X, y = mnist.data / 255., mnist.target
    # X_train, X_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]

    import sklearn.datasets
    dataset = sklearn.datasets.load_digits()
    X_train = dataset.data[:1500]
    X_test = dataset.data[1500:]
    y_train = dataset.target[:1500]
    y_test = dataset.target[1500:]

    network = MLP(input_size=64, output_size=10, hidden_layer_size=[128], batch_size=200, activation="sigmoid",
                  output_layer='softmax', loss='cross_entropy', lr=0.1)

    network.fit(X_train, y_train, 100, True)

    acc = network.score(X_test, y_test)
    print('Test Accuracy: {}'.format(acc))


def sklearn_mlp():
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_mldata
    from sklearn.neural_network import MLPClassifier

    # mnist =fetch_mldata("MNIST original")
    # X, y = mnist.data / 255., mnist.target
    # X_train, X_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]

    import sklearn.datasets
    dataset = sklearn.datasets.load_digits()
    X_train = dataset.data[:1500]
    X_test = dataset.data[1500:]
    y_train = dataset.target[:1500]
    y_test = dataset.target[1500:]

    mlp = MLPClassifier(hidden_layer_sizes=(128), max_iter=100, alpha=1e-4,
                        solver='sgd', activation='logistic', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.01)
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))


def main():
    print('Class 2 Multiple Layer Perceptron (MLP) Example')
    my_mlp()

    print ('')

    print('Class 2 sklearn MLP Example')
    sklearn_mlp()


if __name__ == "__main__":
    main()
