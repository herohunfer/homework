import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer, num_filter, activation=None, BN=False, pooling=False):
    """
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    conv = mx.sym.Convolution(data=input_layer,
                              num_filter=num_filter,
                              kernel=(3,3),
                              stride=(1,1),
                              no_bias=True)

    # Find the doc of mx.sym.Convolution by help command
    if activation is not None:
        conv = mx.sym.Activation(data = conv, act_type=activation)

    # Do you need BatchNorm?
    if BN:
        conv = mx.sym.BatchNorm(conv)

    # Do you need pooling?
    # What is the expected output shape?
    if pooling:
        conv = mx.sym.Pooling(data=conv, stride=(2, 2), kernel=(2, 2), pool_type='max')
    return conv


# Optional
def inception_layer():
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    pass


def get_conv_sym():

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    #data_f = mx.sym.flatten(data=data)
    # How deep the network do you want? like 4 or 5
    l = conv_layer(input_layer=data, num_filter=32, activation="relu", BN=False, pooling=True)
    l = conv_layer(input_layer=l, num_filter=64, activation="relu", BN=False, pooling=True)
    l = conv_layer(input_layer=l, num_filter=128, activation="relu", BN=False, pooling=True)
    # How wide the network do you want? like 32/64/128 kernels per layer
    # MNIST has 10 classes
    fc = mx.sym.FullyConnected(data=l,num_hidden=10, no_bias=True)
    # Softmax with cross entropy loss
    conv = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    # How is the convolution like? Normal CNN? Inception Module? VGG like?
    return conv