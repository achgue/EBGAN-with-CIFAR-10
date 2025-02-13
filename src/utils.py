from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Function to encode class information into one-hot vectors
def encode_class(value):
    x = np.zeros((10))
    x[value] = 1
    return x

# Function to decode one-hot vectors into class labels
def decode_class(value):
    return np.where(value == 1)[0][0]

# Function to generate random noise batches and corresponding class names
def get_random_noise(batch_size, noise_size, num_classes = 10):
    random_noise_batches = np.random.randn(batch_size , noise_size)
    class_array = np.random.choice(10, size=batch_size)
    categorical_array = np.zeros((batch_size, num_classes))
    categorical_array[np.arange(batch_size), class_array] = 1
    return random_noise_batches, categorical_array

# Function to generate fake samples using the generator network
def get_fake_samples(generator_network, batch_size, noise_size, class_name=-1):
    random_noise_batches, class_names = get_random_noise(batch_size, noise_size)
    if class_name != -1:
        class_names = np.array([encode_class(class_name)] * batch_size)
    fake_samples = generator_network.predict_on_batch([random_noise_batches, class_names])
    return [fake_samples, class_names]

# Function to fetch real samples from the training dataset
def get_real_samples(batch_size, trainX, trainY, class_names=-1,):
    random_indices = np.random.choice(len(trainX), size=batch_size)
    if class_names == -1:
        class_names = np.zeros((batch_size, 10))
        class_names [np.arange(batch_size),trainY[random_indices][0]] = 1
    else:
        random_indices = np.concatenate([np.random.choice(np.where(trainY == cls)[0], size=1) for cls in class_names])
    real_images = trainX[random_indices]
    return [real_images, class_names]

# Function to display generated images from the generator network
def show_generator_results(generator_network, noise_size):
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for k in range(10):
        fake_samples = get_fake_samples(generator_network, 10, noise_size, k)
        for j in range(10):
            axs[k][j].imshow((fake_samples[0][j] + 1) / 2)
            axs[k][j].axis('off')
    plt.show()