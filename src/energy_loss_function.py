import tensorflow.keras.backend as K
import tensorflow as tf

# Energy-based loss (L2 reconstruction error)
def energy_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        regularization = 0.01 * tf.reduce_mean(tf.square(y_pred))
        return mse + regularization