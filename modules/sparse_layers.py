import tensorflow as tf
import numpy as np

class SparseConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, weightSparcity=0.5, strides=(1,1), padding="SAME",
            dilation_rate=(1,1), use_bias=True, use_batch_norm=False):
        
        super(SparseConv2D, self).__init__(filters, kernel_size, strides=strides,
                padding= padding, dilation_rate=dilation_rate, use_bias=use_bias)

        assert 0 < weightSparcity < 1
        self.sparsity = weightSparcity



    #def build(self, input_shape):
    #    super(SparseConv2D, self).build(input_shape)
    #    #self.updateWeights()

    def updateWeights(self):
        K = self.kernel
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
        self.kernel = K*w

    def call(self, inputs, training=True):
        if training:
            self.updateWeights()
        return super(SparseConv2D, self).call(inputs)


class SparseDense(tf.keras.layers.Dense):
    def __init__(self, units, weightSparcity=0.5, use_bias=True):
        super(SparseDense, self).__init__(units, use_bias=use_bias)
        assert 0<weightSparcity<1
        self.sparsity = weightSparcity

    #def build(self, input_shape):
    #    if not self.module.built:
    #        self.module.build(input_shape)

    #    self.updateWeights()
    #    super(SparseWeights1D, self).build(input_shape)

    def updateWeights(self):
        K = self.kernel
        in_units, out_units = K.shape
        #Total number of zeros in the filter
        numZeros = int(round((1.0-self.sparsity)*in_units))
        outputIndices = np.arange(out_units)
        inputIndices = np.array([np.random.permutation(in_units)[:numZeros] for _ in outputIndices], dtype=np.long)
        inputIndices = inputIndices.transpose()
        w = np.ones(K.shape)
        w[inputIndices, outputIndices] = 0.0

        self.kernel = K*w

    def call(self, inputs, training=True):
        if training:
            self.updateWeights()
        return super(SparseDense, self).call(inputs)
