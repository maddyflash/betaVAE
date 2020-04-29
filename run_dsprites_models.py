"""
This is the file to run from command line to train and output models initially 
"""
# --------import functions---------
#imports functions to download and work with dsprites
import dsprites as Ds 
#from dsprites import imshow,  make_grid

# import VAE criterion (train_model(model, epochs, train, optimiser, beta, lhd='bernoulli'):
import bvae

# import functions necessary for this file
import tensorflow as tf
import sys
import numpy as np


# helper functions
# system check functions
def test_tf(): 
    """used stack overflow and tf docs to learn about how to access versions and
     make sure that tf was set up properly >>
     version - https://stackoverflow.com/questions/1252163/printing-python-version-in-output
     tf version - https://stackoverflow.com/questions/38549253/how-to-find-which-version-of-tensorflow-is-installed-in-my-system
     tf set up correct - https://www.tensorflow.org/api_docs/python/tf/test/is_gpu_available
    """
    ''' tests that tf and gpu are loaded properly'''
    print(sys.version)
    print(tf.__version__)
    print(tf.test.is_gpu_available())
    
def download_data():
    ds = Ds.DSprites(download=True)
    return ds
    
def cast_dtype(x):
    return tf.cast(x, tf.float32)
    
# -----------------------------------------------------------------------------
# Build this VAE 
class VAE(tf.keras.Model): #this makes a VAE model. it does not specify the criterion.
# its like the picture of the nodes, but not specifying how to train accross it
# (from code in master or "harry" branch)
    def __init__(self, latent_dim=10):
        super(VAE, self).__init__()
        self._latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((64, 64, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1200, activation='relu'),
            tf.keras.layers.Dense(1200, activation='relu'),
            tf.keras.layers.Dense(latent_dim * 2) # why output 20?
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(latent_dim), #how input 10 if last output 20?
            tf.keras.layers.Dense(1200, activation='tanh'),
            tf.keras.layers.Dense(1200, activation='tanh'),
            tf.keras.layers.Dense(1200, activation='tanh'),
            tf.keras.layers.Dense(4096),
            tf.keras.layers.Reshape((64, 64, 1))
        ])

    def call(self, x): raise NotImplementedError

    def encode(self, x): # should encode a given image according to trained model
    # this should return the latents
        h = self.encoder(x) #puts image into encoder; returns...?
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        return self.reparameterise(mean, logvar), mean, logvar
    
    def decode(self, z): # gives image from latent vector 
        return self.decoder(z)

    @staticmethod
    def reparameterise(mean, logvar): #??
        # log sig^2 = 2 log sig => exp(1/2 log sig^2) = exp(log sig) = sig
        eps = tf.random.normal(mean.shape, mean=0.0, stddev=1.0)
        return mean + tf.exp(logvar * 0.5) * eps

# train models
def train_models(beta_val, EPOCHS, LATENT_DIM, BATCH_SIZE, TR_SIZE):
    model = VAE(latent_dim=LATENT_DIM) #initialize model
    train = tf.data.Dataset.from_tensor_slices(ds.subset(size=TR_SIZE)) #create training set?
    train = (train
         .map(cast_dtype) # not letting me cast on import in dsprites class
         .shuffle(2**10)
         .batch(BATCH_SIZE)) #preprocess training set?
    losses = bvae.train_model(model, EPOCHS,
                      train, tf.keras.optimizers.Adagrad(learning_rate=1e-2),
                      beta=beta_val, lhd='bernoulli')
    return model, losses



# output models




if __name__ == "__main__":
    print("hello world; this is runmodels.py")
    #test_tf()
# used tf docs to learn how to save and load models
# https://www.tensorflow.org/tutorials/keras/save_and_load
    ds = download_data()
    EPOCHS = 100
    LATENT_DIM = 10
    BATCH_SIZE = 32
    TR_SIZE = 550_000
    #beta_val = 4

    for beta_val in range(1,11,1):
        model, loss = train_models((beta_val), EPOCHS, LATENT_DIM, BATCH_SIZE, TR_SIZE)
        nploss = np.asarray(loss, dtype=np.float32)
        np.savetxt("vae"+str(beta_val)+"_losses_100e_550ktr.txt", nploss)
        model.save_weights("vae"+str(beta_val)+"_weights_100e_550ktr.h5")
        print("vae"+str(beta_val)+" training complete")

    print("complete")
    
    beta_val=0.5
    model, loss = train_models((beta_val), EPOCHS, LATENT_DIM, BATCH_SIZE, TR_SIZE)
    nploss = np.asarray(loss, dtype=np.float32)
    np.savetxt("vae"+str(beta_val)+"_losses_100e_550ktr.txt", nploss)
    model.save_weights("vae"+str(beta_val)+"_weights_100e_550ktr.h5")
    print("vae"+str(beta_val)+" training complete")
    
    print("complete")
    
