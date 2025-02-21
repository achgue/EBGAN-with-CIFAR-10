from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from src.utils import show_generator_results, get_fake_samples, get_real_samples, get_random_noise, plot_training_history
from discriminator import Discriminator
from generator import Generator
from gan_assembler import assemble_model


# Load CIFAR-10 dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()

# Display shapes of training and testing data
print('Training data shapes: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Testing data shapes: X=%s, y=%s' % (testX.shape, testY.shape))

# Display a grid of 7x7 random images from the training set
plt.figure(figsize=(7, 7))
for k in range(7):
    for j in range(7):
        i = np.random.randint(0, 10000)
        plt.subplot(7, 7, k * 7 + j + 1)
        plt.imshow(trainX[i], cmap='gray_r')
        plt.axis('off')
plt.show()

unique_classes = np.unique(trainY)

# Normalize pixel values to the range [-1, 1]
trainX = [(image - 127.5) / 127.5 for image in trainX]
testX = [(image - 127.5) / 127.5 for image in testX]

# Reshape the data to match the input shape expected by the model
trainX = np.reshape(trainX, (50000, 32, 32, 3))
testX = np.reshape(testX, (10000, 32, 32, 3))

# Display the shapes of the training and testing datasets
print(f"Training data shape: {trainX.shape}")
print(f"Testing data shape: {testX.shape}")
print(f"Training labels shape: {trainY.shape}")
print(f"Testing labels shape: {testY.shape}")

discriminator = Discriminator()
generator = Generator()
ebgan = assemble_model(discriminator, generator)


# Set hyperparameters
epochs = 200
batch_size = 256
steps = len(trainX) // batch_size
noise_size = 100

# Variables to store the loss and accuracy for plotting
d_losses = []
g_losses = []
d_accs = []

# Training loop
for i in range(epochs):
    # Display generated images every 5 epochs
    if i % 5 == 0:
        show_generator_results(generator.network, noise_size)
    # Iterate over training steps
    for j in range(steps):
        # Generate fake and real samples
        fake_samples = get_fake_samples(generator.network, batch_size // 2, noise_size)
        real_samples = get_real_samples(trainX= trainX, trainY=trainY, batch_size=batch_size // 2)
        # Create labels for fake and real samples
        fake_y = np.zeros((batch_size // 2, 1))
        real_y = np.ones((batch_size // 2, 1))
        # Combine fake and real samples and labels
        input_batch_part1 = np.vstack((fake_samples[0], real_samples[0]))
        input_batch_part2 = np.vstack((fake_samples[1], real_samples[1]))
        input_batch_final = [input_batch_part1, input_batch_part2]
        output_labels = np.vstack((fake_y, real_y))
        # Update Discriminator weights
        discriminator.network.trainable = True
        loss_d = discriminator.network.train_on_batch(input_batch_final, output_labels)
        # Generate random noise and class values for GAN input
        noise_batches, class_values = get_random_noise(batch_size, noise_size)
        gan_input = [noise_batches, class_values]
        # Make the Discriminator believe these are real samples and calculate loss to train the generator
        gan_output = np.ones((batch_size))
        # Update Generator weights
        discriminator.network.trainable = False
        loss_g = ebgan.train_on_batch(gan_input, gan_output)
        
    # Store the losses and accuracies
    d_losses.append(loss_d[0])
    g_losses.append(loss_g)
    d_accs.append(loss_d[1] * 100)
    # Display training progress every epoch
    print(f"Epoch: {i}, D-Loss: {loss_d[0]:.3f}, D-Acc: {loss_d[1]*100:.3f}, G-Loss: {loss_g:.3f}")
    
# Call the function to plot the training history
plot_training_history(d_losses, g_losses, d_accs)
    
# Generate and display unlimited samples using the generator network
# This loop generates and shows generated images twice
for i in range(2):
    show_generator_results(generator.network)
    print("-" * 100)