# Import necessary layers and model from TensorFlow
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class Generator():

    # Define input layers for random noise and class information
    random_input = Input(shape=(100,))
    class_input = Input(shape=(10,))

    # Random Input Processing
    x1 = Dense(8 * 8 * 128)(random_input)  # Start with 8x8 resolution
    x1 = Activation('relu')(x1)
    x1 = Reshape((8, 8, 128))(x1)

    # Class Input Processing
    x2 = Dense(8 * 8)(class_input)
    x2 = Activation('relu')(x2)
    x2 = Reshape((8, 8, 1))(x2)  # Match spatial dimensions with x1

    # Concatenate processed inputs
    x = Concatenate()([x1, x2])

    # Deconvolutional Layers
    x = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding="same")(x)  # Upsample to (16,16,128)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding="same")(x)  # Upsample to (32,32,64)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.8)(x)

    # Final layer to match CIFAR-10 image size (32,32,3)
    x = Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding="same")(x)  # Keep (32,32,3)
    generated_image = Activation('tanh')(x)  # Normalize to [-1,1] for GAN training

    # Create and display the architecture of the generator network
    network = Model(inputs=[random_input, class_input], outputs=generated_image)
    network.summary()