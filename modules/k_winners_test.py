import tensorflow as tf
import numpy as np
from k_winners import KWinners


class KWinnersTest(tf.test.TestCase):
    def test_k_winners_2d(self):

        kwargs = {
            'filters': 30,
            'kernel_size': 3,
            'padding': 'same',
            'kernel_regularizer': 'l2',
            'bias_regularizer': 'l2',
            'activity_regularizer': 'l2',
            'strides': 1
        }

        with self.cached_session(use_gpu=True):
            x = tf.Variable(tf.random.uniform((1,28,28,1)), dtype=tf.float32)
            conv = tf.keras.layers.Conv2D(**kwargs)(x)
            out = KWinners(30,400)(conv)
            expected_zero_frac = 1-400.0/(28*28*30)
            self.assertEqual(tf.math.zero_fraction(out), expected_zero_frac)

    def test_k_winners_1d(self):

        with self.cached_session(use_gpu=True):

            x = tf.Variable(np.random.uniform(low=0, high=1, size=(5,10)), dtype='float32')

            linear = tf.keras.layers.Dense(20)(x)
            out = KWinners(20,10)(linear) 
            self.assertEqual(tf.math.zero_fraction(out), 0.5)



if __name__=="__main__":
    tf.test.main()

