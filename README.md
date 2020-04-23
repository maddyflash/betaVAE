# betaVAE
Group 5 [beta VAE](https://openreview.net/forum?id=Sy2fzU9gl) paper reproduction assignment.

# Data

The weights and data used in this experiment can be downloaded from the [releases](https://github.com/maddyflash/betaVAE/releases)
page. Place the downloaded weights in to the data folder of this repository and unzip them: the scripts expect to find
these weights there.

# Reproducing results

A brief description about this repository:


# Notebooks & Scripts

Name                              | Content                                                               | Used in
----------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------
beta\_finder.ipynb                |  Evaluate beta = 0.5, 1, ... 10                                       | Disentanglement score plot in section 3
beta\_vae\_metric.ipynb           |  Evaluate beta = 4, PCA, ICA, VAE, VAE-untrained                      | Disentanglement score table in section 3, latent activation plot in section 3, and discussion in section 4.2
factor\_vae\_metric.ipynb         |  Similar to beta\_vae\_metric.ipynb, but with Factor-VAE's metric     | Disentanglement score table & plot in section 3.
dSprites\_proper\_training.ipynb  |  Train beta-VAE model on dSprites for disentanglement evaluation      | Train and save weights for use in beta\_vae\_metric.ipynb and factor\_vae\_metric.ipynb
