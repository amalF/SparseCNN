import tensorflow as tf

class DenseNet(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(30, (3,3))
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2,
                                                strides=2,
                                                padding="VALID")
        self.act1 = tf.keras.layers.ReLU()

        self.fc2 = tf.keras.layers.Dense(150, activation="relu")
        self.fc3 = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=True):
        x = self.conv1(inputs, training=training)
        x = self.pool1(x) 
        x = self.act1(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc2(x, training=training)
        logits = self.fc3(x)
        return logits

