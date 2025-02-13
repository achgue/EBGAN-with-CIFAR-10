
import tensorflow
from energy_loss_function import energy_loss


def assemble_model(discriminator, generator):
    # Import necessary optimizer from TensorFlow
    adam_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

    # Compile the discriminator network with binary crossentropy loss and Adam optimizer
    #discriminator_network.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    discriminator.network.compile(loss=energy_loss, optimizer=adam_optimizer, metrics=['accuracy'])

    # Set the discriminator to be non-trainable during the training of the GAN
    discriminator.network.trainable = False

    # Generate images using the generator network based on random noise and class input
    gan_input = generator.network([generator.random_input, generator.class_input])

    # Pass the generated images and class input through the discriminator to get GAN output
    gan_output = discriminator.network([gan_input, generator.class_input])

    # Create the GAN model by connecting the generator and discriminator
    gan_model = tensorflow.keras.models.Model([generator.random_input, generator.class_input], gan_output)

    # Display the architecture of the GAN model
    gan_model.summary()

    # Import necessary optimizer from TensorFlow
    adam_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

    # Compile the GAN model with binary crossentropy loss and the configured Adam optimizer
    gan_model.compile(loss=energy_loss, optimizer=adam_optimizer)
    
    return gan_model