import tensorflow as tf
import numpy as np
from sparse_weights import *


class SparseWeightsTest(tf.test.TestCase):
    def test_sparse_weights_2d(self):

        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'padding': 'valid',
            'kernel_regularizer': 'l2',
            'bias_regularizer': 'l2',
            'activity_regularizer': 'l2',
            'strides': 1
        }

        with self.cached_session(use_gpu=True):
            x = tf.Variable(np.ones((1,5,5,2)), dtype=tf.float32)
            conv = tf.keras.layers.Conv2D(**kwargs)
            conv.build((None,5,5,2))
            SparseWeights2D(conv,0.5).build((None, 5, 5, 2))
            self.assertEqual(tf.math.zero_fraction(conv.kernel), 0.5)

    def test_sparse_weights_1d(self):

        with self.cached_session(use_gpu=True):

            x = tf.Variable(np.random.randint(low=0, high=7, size=(10)), dtype='float32')

            linear = tf.keras.layers.Dense(5)
            linear.build((10))
            SparseWeights1D(linear,0.5).build((10))
            self.assertEqual(tf.math.zero_fraction(linear.kernel), 0.5)

    def test_model(self):
        kwargs = {
            'filters': 3,
            'kernel_size': 3,
            'padding': 'same',
            'strides': 1
        }

        x = tf.Variable(np.random.uniform(low=0, high=1, size=(1,5,5,3)), dtype='float32')
        cnn = tf.keras.layers.Conv2D(use_bias=False,**kwargs)
        sparseCnn = SparseWeights2D(cnn, 0.5)
        linear = tf.keras.layers.Dense(10)
        sparselinear = SparseWeights1D(linear, 0.5)

        outsparsecnn = sparseCnn(x)

        out = tf.keras.layers.Flatten()(outsparsecnn)
        outlinear = sparselinear(out)

        expected_conv_output = tf.nn.convolution(x,cnn.kernel, padding='SAME', strides=1)

        self.assertAllEqual(outsparsecnn, expected_conv_output)
        self.assertAllEqual(outlinear, tf.matmul(out, linear.kernel))


if __name__=="__main__":
    tf.test.main()
