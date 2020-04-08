'''train model function for the beta vae trining criterion

Reorganized with very slight modifications from code in "harry" or master branch'''

import tqdm
import tensorflow as tf
import numpy as np

# training functions
# define training function with beta_vae criterion
def criterion(x, x_recon, mean, logvar, beta=1.0, lhd='bernoulli'):
    """
    x - original image
    x_recon - LOGITS! depending on `lhd`, it'll either be activated by sigmoid
              or a normal distribution
    """
    kl = -0.5 * tf.reduce_sum(1 + logvar - mean**2 - tf.exp(logvar))
    if lhd.lower() == 'bernoulli':
        lh = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x))
    elif lhd.lower() == 'normal':
        # TODO!
        raise NotImplementedError
    else:
        raise ValueError(f"Expected lhd to be one of bernoulli or normal, got {lhd}.")
    return (lh + beta * kl) / x.shape[0]

def get_step_function():
    @tf.function
    def step(model, x, optimiser, beta, lhd):
        with tf.GradientTape() as tape:
            z, mean, logvar = model.encode(x)
            x_recon = model.decode(z)
            loss = criterion(x, x_recon, mean, logvar, beta, lhd)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    return step
    
def train_model(model, epochs, train, optimiser, beta, lhd='bernoulli'):
    """ trains a given model according to bvae criterion
    :param model: the tf model object
    :param epochs: number of epochs
    :param train: the training set
    :param optimiser: which keras optimizer ex. tf.keras.optimizers.Adarad(learning_rate=1e-2)
    :param beta: the value of beta
    :param lhd: ='bernoulli' currently only option
    
    :return: losses as array; trains the model (updates weights) 
    """
    # redefine step function because it's a compiled graph (bug)
    # see: https://github.com/tensorflow/tensorflow/issues/27120
    step = get_step_function()
    rtqdm = tqdm.trange(epochs)
    losses = []
    for e in rtqdm:
        epochs_losses = []
        for x in train:
            epochs_losses.append(step(model, x, optimiser, beta, lhd))
        losses.append(np.mean(epochs_losses))
        rtqdm.set_postfix(loss=losses[-1])
    return losses


if __name__ == "__main__":
    print("hello world; this is bvae.py")
    
    print("complete")
