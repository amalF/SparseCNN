import tensorflow as tf
import numpy as np
from sparse_layers import SparseConv2D, SparseDense


class SparseConv2DTest(tf.test.TestCase):
    def test_sparse_conv_2d(self):

        kwargs = {
            'filters': 32,
            'kernel_size': 3,
            'padding': 'valid',
            'strides': 1,
            'weightSparcity': 0.8
        }

        with self.cached_session(use_gpu=True):
            x = tf.Variable(tf.random.uniform((1,2,2,16)), dtype=tf.float32)
            conv = SparseConv2D(**kwargs)
            out = conv(x)
            self.assertAlmostEqual(round(tf.math.zero_fraction(conv.kernel).numpy(),1), 0.2)


    def test_sparse_dense(self):

        with self.cached_session(use_gpu=True):

            x = tf.Variable(np.random.uniform(size=(10,10)), dtype='float32')

            linear = SparseDense(5)
            out = linear(x)
            self.assertEqual(round(tf.math.zero_fraction(linear.kernel).numpy(),1), 0.5)

if __name__=="__main__":
    tf.test.main()
