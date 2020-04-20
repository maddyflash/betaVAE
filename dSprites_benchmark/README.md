# Notebooks

Most of the notebooks here are generated for section 3's latent activation plots and
section 4's discussion. Mainly, beta/factor\_vae computes the disentanglement score table showed
in section 3. beta\_finder was used to plot the disentaglement scores of all beta=0.5, 1, ... 10 values.
The plots and numbers supporting the main arugments in section 4's discussion can be found in beta\_vae\_metric.

Name                            | Content
------------------------------------------------------------------------------------------------------------
beta\_finder.ipynb                |  Evaluate beta=0.5, 1, ... 10 (generated plot in section 3).
beta\_vae\_metric.ipynb           |  Evaluate beta=4, PCA, ICA, VAE, VAE-untrained (table in section 3).
dSprites\_proper\_training.ipynb  |  Train beta-VAE model on dSprites for disentanglement evaluation.
factor\_vae\_metric.ipynb         |  Similar to beta\_vae\_metric.ipynb, but using Factor-VAE's metric instead.
