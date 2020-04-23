# betaVAE
Group 5 [beta VAE](https://openreview.net/forum?id=Sy2fzU9gl) paper reproduction assignment.

# Data

The weights and data used in this experiment can be downloaded from the [releases](https://github.com/maddyflash/betaVAE/releases)
page. Place the downloaded weights into the data folder of this repository and unzip them: the scripts expect to find
these weights there.

# Reproducing results

The scripts and notebooks to regenerate the results are provided in this repository in the form of Python scripts and notebooks. Most of
the generated data can be found in the data folder except for the trained weights, which can be downloaded [here](https://github.com/maddyflash/betaVAE/releases).

We recommend using conda to re-create our environment by running the following command:

`conda create -n <env-name> --file requirements.txt`


# Notebooks & Scripts

Name                              | Content                                                               | Used in
----------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------
beta\_finder.ipynb                |  Evaluate beta = 0.5, 1, ... 10                                       | Disentanglement score plot in section 3
beta\_vae\_metric.ipynb           |  Evaluate beta = 4, PCA, ICA, VAE, VAE-untrained                      | Disentanglement score table in section 3, latent activation plot in section 3, and discussion in section 4.2
factor\_vae\_metric.ipynb         |  Similar to beta\_vae\_metric.ipynb, but with Factor-VAE's metric     | Disentanglement score table & plot in section 3.
dSprites\_proper\_training.ipynb  |  Train beta-VAE model on dSprites for disentanglement evaluation      | Train and save weights for use in beta\_vae\_metric.ipynb and factor\_vae\_metric.ipynb
