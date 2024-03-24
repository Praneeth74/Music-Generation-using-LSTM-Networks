import tensorflow as tf
@tf.keras.utils.register_keras_serializable()
class MyModel(tf.keras.Model):
    def __init__(self, input_shape_, input_units, output_units, **kwargs):
        """
            input_shape: A tuple length to representing (time_steps, features)
            input_units: number of units in the first layer (lstm layer)
            output_units: number of units in the output layer
        """
        super(MyModel, self).__init__()
        self.input_shape_ = input_shape_
        self.input_units = input_units
        self.output_units = output_units

        # Layers
        self.lstm1 = tf.keras.layers.LSTM(self.input_units, input_shape=self.input_shape_, return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(512)
        self.batchNorm1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(448)
        self.batchNorm2 = tf.keras.layers.BatchNormalization()
        self.dense_output = tf.keras.layers.Dense(self.output_units, activation='softmax')

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dense1(x)
        x = self.batchNorm1(x, training=training)
        x = self.dense2(x)
        x = self.batchNorm2(x, training=training)
        x = self.dense_output(x)
        return x
    
    def get_config(self):
        config = super(MyModel, self).get_config()
        config.update({
            "input_shape_":self.input_shape_,
            "input_units":self.input_units,
            "output_units":self.output_units
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
