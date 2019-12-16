import tensorflow as tf

from modules import sparse_layers as sp
from modules import k_winners as kw

class SparseNet(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(SparseNet, self).__init__()

        self.conv1 = sp.SparseConv2D(30, (3,3), weightSparcity=1.0)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2,
                                                strides=2,
                                                padding="VALID")
        self.act1 = kw.KWinners(30,400,boostStrength=1.5,boostStrengthFactor=0.85)

        self.fc2 = sp.SparseDense(150, weightSparcity=0.3)
        self.act2 = kw.KWinners(150,50,boostStrength=1.5,boostStrengthFactor=0.85)
        self.fc3 = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=True):
        x = self.conv1(inputs, training=training)
        x = self.pool1(x) 
        x = self.act1(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc2(x, training=training)
        x = self.act2(x, training=training)
        logits = self.fc3(x)
        return logits

