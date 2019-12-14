import tensorflow as tf
import numpy as np


class KWinners(tf.keras.layers.Layer):

    def __init__(self, channels, k, kInferenceFactor=1.0, boostStrength=1.0, boostStrengthFactor=1.0, dutyCyclePeriod=1000):
        super(KWinners, self).__init__()
        self.channels = channels
        self.k = k
        self.kInferenceFactor = kInferenceFactor
        self.boostStrength = boostStrength
        self.boostStrengthFactor = boostStrengthFactor
        self.dutyCyclePeriod = dutyCyclePeriod
        self.dutyCycle = None
        self.dutyCycle = tf.zeros((self.channels))
        self.learningIterations = 0

    def call(self, inputs, training=True):
        total_nrof_neurons = self.channels
        if inputs.shape.ndims>2:
            total_nrof_neurons = total_nrof_neurons*inputs.shape[1]*inputs.shape[2]
            
        if self.k >=total_nrof_neurons:
            return tf.keras.layers.ReLU()(inputs)

        if not training:

            self.k = min(int(self.kInferenceFactor*self.k), total_nrof_neurons)

        x = self.get_kwinners(inputs) 

        if training:
            self._updateDutyCycle(x)

        return x

    def get_kwinners(self, inputs):

        @tf.custom_gradient
        def _get_kwinners(inputs):
        
            boosted = inputs 
            input_shape = inputs.get_shape().as_list()

            if len(input_shape)>3:
                num_units = np.prod(input_shape[1:])
            else:
                num_units = input_shape[-1]
        
            if self.boostStrength != 0.0:
                # target duty ccyle is the percentage of active units
                target_duty_cycle = float(self.k)/num_units
                boostFactors = tf.exp((target_duty_cycle-self.dutyCycle)*self.boostStrength)
                boosted = inputs*boostFactors
         
            if len(boosted.shape)>3:
                boosted = tf.reshape(boosted, (input_shape[0],-1))
            boosted_shape = boosted.shape    

            values, indices = tf.math.top_k(boosted, k=self.k, sorted=False)
            indices_grid = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(
                boosted_shape[:-1]) + [self.k])], indexing='ij')
        
            indices_grid = tf.stack(indices_grid[:-1] + [indices], axis=-1)
            full_indices = tf.reshape(indices_grid, [-1, boosted_shape.ndims])
            values = tf.reshape(values, [-1])
        
            mask_st = tf.SparseTensor(indices=tf.cast(
              full_indices, dtype=tf.int64), values=tf.ones_like(values), dense_shape=(input_shape[0],num_units))
            mask = tf.sparse.to_dense(tf.sparse.reorder(mask_st))
            mask = tf.reshape(mask, input_shape)
            result = inputs*mask
        
            def grad_fn(*grad_ys):
                return grad_ys*mask
        
            return result, grad_fn

        return _get_kwinners(inputs)

    def _updateDutyCycle(self,x):
        batch_size = x.shape[0]
        self.learningIterations += batch_size
                                                                            
        period = min(self.dutyCyclePeriod, self.learningIterations)
        self.dutyCycle = tf.multiply(self.dutyCycle, (period - batch_size))

        scaleFactor = 1.0
        ax = [0]
        if len(x.shape)>3:
            scaleFactor = x.shape[1]*x.shape[2]
            ax = tf.range(0,x.shape.ndims-1)

        s = tf.reduce_sum(x, axis=ax)/scaleFactor
        self.dutyCycle = self.dutyCycle +s
        self.dutyCycle = tf.divide(self.dutyCycle, period)































