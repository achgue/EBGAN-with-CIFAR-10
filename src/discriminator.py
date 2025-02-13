# Import necessary layers and model from TensorFlow
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class Discriminator():
    # Define input layers for image and class information
    image_input = Input(shape=(32, 32, 3))
    class_input = Input(shape=(10,))

    # Class Input Processing
    x2 = Dense(32 * 32 * 1)(class_input)  # Match spatial dimensions
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = Reshape((32, 32, 1))(x2)  # Reshape to (32,32,1)

    # Concatenate processed inputs
    x = Concatenate()([image_input, x2])  # Combined shape: (32,32,4)

    # Convolutional Layers
    x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)  # (16,16,128)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)  # (8,8,128)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)  # (4,4,64)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Flatten()(x)  # Flatten to 1D

    # Fully Connected Layers
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(1)(x)  # Output real vs fake score
    real_vs_fake_output = Activation('sigmoid')(x)

    # Create and display the architecture of the discriminator network
    network = Model(inputs=[image_input, class_input], outputs=real_vs_fake_output)
    network.summary()
