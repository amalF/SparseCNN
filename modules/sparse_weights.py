import tensorflow as tf
import numpy as np

class SparseWeights2D(tf.keras.layers.Layer):
    def __init__(self, module, sparsity):
        super(SparseWeights2D, self).__init__()
        self.module = module
        self.sparsity = sparsity

    def build(self, input_shape):
        if not self.module.built:
            self.module.build(input_shape)
        self.updateWeights()
        super(SparseWeights2D, self).build(input_shape)

    def updateWeights(self):
        K = self.module.kernel
        kernel_size1, kernel_size2, in_channels, out_channels = K.shape

        filter_size = kernel_size1*kernel_size2*in_channels
        #Total number of zeros in the filter
        numZeros = int(round((1.0-self.sparsity)*filter_size))

        outputIndices = np.arange(out_channels)
        inputIndices = np.array([np.random.permutation(filter_size)[:numZeros] for _ in outputIndices], dtype=np.long)
        inputIndices = inputIndices.transpose()
        w = np.ones(K.shape).reshape(-1, out_channels)
        w[inputIndices, outputIndices] = 0.0
        w = w.reshape(K.shape)
        self.module.kernel = K*w

    def call(self, inputs, training=True):
        if training:
            self.updateWeights()
        return self.module(inputs)


class SparseWeights1D(tf.keras.layers.Layer):
    def __init__(self, module, sparsity):
        super(SparseWeights1D, self).__init__()
        self.module = module
        self.sparsity = sparsity
    def build(self, input_shape):
        if not self.module.built:
            self.module.build(input_shape)

        self.updateWeights()
        super(SparseWeights1D, self).build(input_shape)

    def updateWeights(self):
        K = self.module.kernel
        in_units, out_units = K.shape
        #Total number of zeros in the filter
        numZeros = int(round((1.0-self.sparsity)*in_units))
        outputIndices = np.arange(out_units)
        inputIndices = np.array([np.random.permutation(in_units)[:numZeros] for _ in outputIndices], dtype=np.long)
        inputIndices = inputIndices.transpose()
        w = np.ones(K.shape)
        w[inputIndices, outputIndices] = 0.0

        self.module.kernel = K*w

    def call(self, inputs, training=True):
        if training:
            self.updateWeights()
        return self.module(inputs)
