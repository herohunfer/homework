import mxnet as mx
import logging

def get_mlp(data):

    l = mx.sym.FullyConnected(data=data, num_hidden=100)
    l = mx.sym.Activation(data=l, act_type="relu")
    l = mx.sym.FullyConnected(data=l, num_hidden=100)
    l = mx.sym.Activation(data=l, act_type="relu")
    l = mx.sym.FullyConnected(data=l, num_hidden=100)
    l = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return l

def get_cnn(data):

    l = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
    l = mx.sym.Activation(data=l, act_type="tanh")
    l = mx.sym.Pooling(data=l, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # second conv layer
    l = mx.sym.Convolution(data=l, kernel=(5, 5), num_filter=50)
    l = mx.sym.Activation(data=l, act_type="tanh")
    l = mx.sym.Pooling(data=l, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # first fullc layer
    l = mx.sym.flatten(data=l)
    l = mx.symbol.FullyConnected(data=l, num_hidden=500)
    l = mx.sym.Activation(data=l, act_type="tanh")
    # second fullc
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # softmax loss
    l = mx.sym.SoftmaxOutput(data=l, name='softmax')

    return l

if __name__ == "__main__":
    # get data
    mnist = mx.test_utils.get_mnist()

    # Setup data iterator
    batch_size = 100
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    data = mx.sym.var('data')

    # get the model
    mlp = get_mlp(data)
    cnn = get_cnn(data)

    print mlp.get_internals()
    print cnn.get_internals()

    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # create a trainable module on CPU
    mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
    mlp_model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  optimizer='sgd',  # use SGD to train
                  optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
                  eval_metric='acc',  # report accuracy during training
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  # output progress for each 100 data batches
                  num_epoch=10)  # train for at most 10 dataset passes

