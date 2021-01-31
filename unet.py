import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal


def _calc_initializer_stddev(filters, kernel_size):
    N = (kernel_size ** 2) * filters
    std = tf.math.sqrt(2 / N)
    return std


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ConvBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size

        self.stddev = _calc_initializer_stddev(filters, kernel_size)
        self.conv1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            kernel_initializer=RandomNormal(stddev=self.stddev)
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            kernel_initializer=RandomNormal(stddev=self.stddev)
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.dropout = layers.Dropout(0.2)


    def call(self, inputs, training=None):
        x = inputs

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        if training:
            x = self.dropout(x)

        return x


    def get_config(self):
        return dict(
            filters=self.filters,
            kernel_size=self.kernel_size,
            stddev=self.stddev,
            **super(ConvBlock, self).get_config(),
        )


def make_unet(height=None,
              width=None,
              channels=3,
              num_classes=1):
    # num_classes=1
    # Because our model classifies object or background

    inputs = Input(shape=(height, width, channels), name='inputs', dtype='float32')
    x = inputs

    feature_maps = {}

    contracting_block_depth0 = ConvBlock(64, 3)
    contracting_block_depth1 = ConvBlock(128, 3)
    contracting_block_depth2 = ConvBlock(256, 3)
    contracting_block_depth3 = ConvBlock(512, 3)
    contracting_block_depth4 = ConvBlock(1024, 3)

    maxpool_depth1 = layers.MaxPooling2D((2, 2))
    maxpool_depth2 = layers.MaxPooling2D((2, 2))
    maxpool_depth3 = layers.MaxPooling2D((2, 2))
    maxpool_depth4 = layers.MaxPooling2D((2, 2))

    upconv_depth4 = layers.Conv2DTranspose(filters=512, kernel_size=(2,2), strides=2)
    upconv_depth3 = layers.Conv2DTranspose(filters=256, kernel_size=(2,2), strides=2)
    upconv_depth2 = layers.Conv2DTranspose(filters=128, kernel_size=(2,2), strides=2)
    upconv_depth1 = layers.Conv2DTranspose(filters=64, kernel_size=(2,2), strides=2)

    expanding_block3 = ConvBlock(512, 3)
    expanding_block2 = ConvBlock(256, 3)
    expanding_block1 = ConvBlock(128, 3)
    expanding_block0 = ConvBlock(64, 3)

    x = contracting_block_depth0(x)
    feature_maps[0] = x

    x = contracting_block_depth1(maxpool_depth1(x))
    feature_maps[1] = x

    x = contracting_block_depth2(maxpool_depth2(x))
    feature_maps[2] = x

    x = contracting_block_depth3(maxpool_depth3(x))
    feature_maps[3] = x

    x = contracting_block_depth4(maxpool_depth4(x))    

    x = upconv_depth4(x) # 1024 -> 512

    x = tf.concat([feature_maps[3], x], axis=-1)
    x = upconv_depth3(expanding_block3(x)) # 512 -> 256

    x = tf.concat([feature_maps[2], x], axis=-1)
    x = upconv_depth2(expanding_block2(x)) # 256 -> 128

    x = tf.concat([feature_maps[1], x], axis=-1)
    x = upconv_depth1(expanding_block1(x)) # 128 -> 64

    x = tf.concat([feature_maps[0], x], axis=-1)
    x = expanding_block0(x)

    stddev = _calc_initializer_stddev(16, 3)
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      strides=1,
                      kernel_initializer=RandomNormal(stddev=stddev),
                      padding='same',
                      activation = 'relu')(x)

#     x = layers.BatchNormalization()(x)

    outputs = layers.Conv2D(filters=num_classes,
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            activation='sigmoid',
                            name='outputs')(x)
    model = Model(inputs, outputs=[outputs], name="unet")

    return model
