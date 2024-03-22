from utils.music_utils import *
import tensorflow
class MyModel(tensorflow.keras.Model):
    def __init__(self, input_shape, input_units, output_units):
        """
            input_shape: A tuple length to representing (time_steps, features)
            input_units: number of units in the first layer (lstm layer)
            output_units: number of units in the output layer
        """
        super(MyModel, self).__init__()
        self.input_shape = input_shape
        self.input_units = input_units
        self.output_units = output_units

        # Layers
        self.lstm1 = LSTM(input_units, input_shape=input_shape, return_sequences=False)
        self.dense1 = Dense(512)
        self.batchNorm1 = BatchNormalization()
        self.dense2 = Dense(448)
        self.batchNorm2 = BatchNormalization()
        self.dense_output = Dense(output_units, activation='softmax')

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dense1(x)
        x = self.batchNorm1(x)
        x = self.dense2(x)
        x = self.batchNorm2(x)
        x = self.dense_output(x)
        return x


