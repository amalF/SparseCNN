import tensorflow as tf
import numpy as np
from k_winners import KWinners


class KWinnersTest(tf.test.TestCase):
    def test_k_winners_2d(self):

        kwargs = {
            'filters': 32,
            'kernel_size': 3,
            'padding': 'valid',
            'kernel_regularizer': 'l2',
            'bias_regularizer': 'l2',
            'activity_regularizer': 'l2',
            'strides': 1
        }

        with self.cached_session(use_gpu=True):
            x = tf.Variable(np.ones((1,32,32,16)), dtype=tf.float32)
            conv = tf.keras.layers.Conv2D(**kwargs)(x)
            out = KWinners(32,10000)(conv)
            self.assertEqual(tf.math.zero_fraction(out), 0.6527778)

    def test_k_winners_1d(self):

        with self.cached_session(use_gpu=True):

            x = tf.Variable(np.random.randint(low=0, high=7, size=(5,10)), dtype='float32')

            linear = tf.keras.layers.Dense(20)(x)
            out = KWinners(20,10)(linear) 
            self.assertEqual(tf.math.zero_fraction(out), 0.5)

    #def test_model(self):
    #    kwargs = {
    #        'filters': 3,
    #        'kernel_size': 3,
    #        'padding': 'same',
    #        'strides': 1
    #    }

    #    x = tf.Variable(np.random.uniform(low=0, high=1, size=(1,5,5,3)), dtype='float32')
    #    cnn = tf.keras.layers.Conv2D(use_bias=False,**kwargs)
    #    sparseCnn = SparseWeights2D(cnn, 0.5)
    #    linear = tf.keras.layers.Dense(10)
    #    sparselinear = SparseWeights1D(linear, 0.5)

    #    outsparsecnn = sparseCnn(x)

    #    out = tf.keras.layers.Flatten()(outsparsecnn)
    #    outlinear = sparselinear(out)

    #    expected_conv_output = tf.nn.convolution(x,cnn.kernel, padding='SAME', strides=1)

    #    self.assertAllEqual(outsparsecnn, expected_conv_output)
    #    self.assertAllEqual(outlinear, tf.matmul(out, linear.kernel))


if __name__=="__main__":
    tf.test.main()
#def test_kwinners():
#    x = tf.Variable(np.random.uniform(low=0, high=1, size=(5,10)), dtype='float32')
#    with tf.GradientTape() as g:
#        g.watch(x)
#        result = KWinners(10,5)(x)
#        
#    print(result.numpy)
#    print(tf.math.zero_fraction(result))
#    print(g.gradient(result, x))
#
#       
#def test_kwinners2d():
#    x = tf.Variable(np.random.uniform(low=0, high=1, size=(2,1,2,5)), dtype='float32')
#    with tf.GradientTape() as g:
#        g.watch(x)
#        result = KWinners(5,2)(x)
#
#    #print(result.numpy)
#    print(tf.math.zero_fraction(result))
#    print(tf.math.zero_fraction(g.gradient(result, x)))
#
#
#
#if __name__=="__main__":
#    test_kwinners2d()


