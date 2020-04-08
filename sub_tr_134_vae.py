"""
This is the file to run from command line to train and output models
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

# save vae models and txt file strings as variables
v = ["vae0.5_weights_100e_550ktr.h5","vae1_weights_100e_550ktr.h5",
    "vae2_weights_100e_550ktr.h5", "vae3_weights_100e_550ktr.h5",
    "vae4_weights_100e_550ktr.h5", "vae5_weights_100e_550ktr.h5",
    "vae6_weights_100e_550ktr.h5", "vae7_weights_100e_550ktr.h5",
    "vae8_weights_100e_550ktr.h5", "vae9_weights_100e_550ktr.h5",
     "vae10_weights_100e_550ktr.h5"]
vl = ["vae0.5_losses_100e_550ktr.txt","vae1_losses_100e_550ktr.txt",
    "vae2_losses_100e_550ktr.txt", "vae3_losses_100e_550ktr.txt",
    "vae4_losses_100e_550ktr.txt", "vae5_losses_100e_550ktr.txt",
    "vae6_losses_100e_550ktr.txt", "vae7_losses_100e_550ktr.txt",
    "vae8_losses_100e_550ktr.txt", "vae9_losses_100e_550ktr.txt",
    "vae10_losses_100e_550ktr.txt"]

# helper functions
# system check functions
def test_tf(): # see notes in "run_dsprites_models.py" for comments on using 
	# stackoverflow and tf documentation to learn about how to check 
	# python and tf version and tf set up properly
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
# from code in "harry" or master branch
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
    """Used to do additional training on preexisting models by loading in old weights"""
    model = VAE(latent_dim=LATENT_DIM) #initialize model
    # load weights
    if beta_val == 0.5:
        model.load_weights(v[0])
    else:
        model.load_weights(v[beta_val])
    #train
    train = tf.data.Dataset.from_tensor_slices(ds.subset(size=TR_SIZE)) #create training set?
    train = (train
         .map(cast_dtype) # not letting me cast on import in dsprites class
         .shuffle(2**10)
         .batch(BATCH_SIZE)) #preprocess training set?
    losses = bvae.train_model(model, EPOCHS,
                      train, tf.keras.optimizers.Adagrad(learning_rate=1e-2),
                      beta=beta_val, lhd='bernoulli')
    return model, losses

def train_models_new(beta_val, EPOCHS, LATENT_DIM, BATCH_SIZE, TR_SIZE):
    """ Used for training randomizing initial weights for training new models and replicates"""
    model = VAE(latent_dim=LATENT_DIM) #initialize model
    #train
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
    print("hello world; this is going to train more models with sub_tr_134_vae.py")
    #test_tf()
# used tf docs to learn how to save and load models
# https://www.tensorflow.org/tutorials/keras/save_and_load
    ds = download_data()
    EPOCHS = 100 #set to 200 for training 200 additional epochs on original first round models
    LATENT_DIM = 10
    BATCH_SIZE = 32
    TR_SIZE = 550_000
    #beta_val = 4
    
    ctr = 1
    for beta_val in [5, 5, 5, 5, 5]: # this was used to train the replicates for beta=5
        model, loss = train_models_new((beta_val), EPOCHS, LATENT_DIM, BATCH_SIZE, TR_SIZE)
        nploss = np.asarray(loss, dtype=np.float32)
        np.savetxt("vae"+str(beta_val)+ "_rep_" +str(ctr)+"_losses_0-100e_550ktr_rep_2.txt", nploss)
        model.save_weights("vae"+str(beta_val)+ "_rep_" +str(ctr)+"_weights_0-100e_550ktr_rep_2.h5")
        print("vae"+str(beta_val) + "rep "+ str(ctr)+ "training complete")
        ctr += 1

    print("complete")
    """
    ctr = 1
    for beta_val in [1, 3, 4, 5, 7]: # this was used for additional trianing of 1,3,4,5,7
        model, loss = train_models((beta_val), EPOCHS, LATENT_DIM, BATCH_SIZE, TR_SIZE)
        nploss = np.asarray(loss, dtype=np.float32)
        np.savetxt("vae"+str(beta_val)+"_losses_100-300e_550ktr_sup.txt", nploss)
        model.save_weights("vae"+str(beta_val)+"_weights_100-300e_550ktr_sup.h5")
        print("vae"+str(beta_val) + "rep "+ str(ctr)+ "training complete")
        ctr += 1

    print("complete")
    """
    # -----------------------------------------------------------------------------------
    # section below is irrelevant 
    '''beta_val=0.5
    model, loss = train_models((beta_val), EPOCHS, LATENT_DIM, BATCH_SIZE, TR_SIZE)
    nploss = np.asarray(loss, dtype=np.float32)
    np.savetxt("vae"+str(beta_val)+"_losses_100e_550ktr.txt", nploss)
    model.save_weights("vae"+str(beta_val)+"_weights_100e_550ktr.h5")
    print("vae"+str(beta_val)+" training complete")
    
    print("complete")
    '''
