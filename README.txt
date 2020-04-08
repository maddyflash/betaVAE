Posted to GIT is a subset of the COMPLETE working directory detailed in the second section. If you require access to any of the files discussed in the COMPLETE Working Directory README, please contact me.

-------------------------------------------------------------------------------
-------------GIT Working Directory README--------------------------------------
-------------------------------------------------------------------------------

All code files from the full directory have been posted. 

This branch deals with the following tasks:
 1) weights and losses for 100 epochs of beta=0.5 and beta=1-10 (inclusive)
 2) weights and losses for additional 200 epochs of beta=1,3,4,5,7
 3) weights and losses for 5 new, randomized, models of beta=5 for 100 epochs each
 4) loss curves for items 1 and 2 above using the .txt files
 5) reconstruction and latent traversal images for 1 and 2 above
 6) dSprites image samples (also used for reconstruction comparison)
 7) plots with error bars charting the quantitative scores for items 2 and 3 above.

 - run_dsprites_models.py -> item 1
 - sub_tr_134_vae.py -> item 2,3
 - dsprites_analysis.ipynb and dsprites_analysis_sup_clean.ipynb -> item 4,5,6
 - dsprites_analysis_bar_charts.ipynb -> item 7
 - PACKAGES AND VERSIONs -> tf2.1.0_gpu_env_package-list.txt

Note: If using anaconda, my environment can be replicated using the following command:
>> conda create -n <env-name> --file tf2.1.0_gpu_env_package-list.txt
	(source: conda documentation: https://docs.conda.io/projects/conda/en/latest/commands/list.html)

Please let me know if you have any questions or concerns regarding these files. Thank you for your time and consideration.


-------------------------------------------------------------------------------
-------------COMPLETE Working Directory README---------------------------------
-------------------------------------------------------------------------------

Summary:
 - See enumerated items for tasks addressed in directory
 - run_dsprites_models.py -> item 1
 - sub_tr_134_vae.py -> item 2,3
 - dsprites_analysis.ipynb and dsprites_analysis_sup_clean.ipynb -> item 4,5,6
 - dsprites_analysis_bar_charts.ipynb -> item 7
 - PACKAGES AND VERSIONs -> tf2.1.0_gpu_env_package-list.txt
 - "dsprites_ndarray_co1sh"..."x64.npz" zip file is the dSprites data published by Deep Mind exactly as found here: https://github.com/deepmind/dsprites-dataset 


This is the directory with files related to the qualitative analysis for learning the dSprites dataset using beta-VAE. The motivation for these experiments was to reproduce the beta-vae results published by Higgins et. al. in 2017. Specifically, the files in this folder were used to generate (for dsprites):
 1) weights and losses for 100 epochs of beta=0.5 and beta=1-10 (inclusive)
 2) weights and losses for additional 200 epochs of beta=1,3,4,5,7
 3) weights and losses for 5 new, randomized, models of beta=5 for 100 epochs each
 4) loss curves for items 1 and 2 above using the .txt files
 5) reconstruction and latent traversal images for 1 and 2 above
 6) dSprites image samples (also used for reconstruction comparison)
 7) plots with error bars charting the quantitative scores for items 2 and 3 above.

Note: Harry did the plots with error bars for item 1 above. Harry also computed the quantitative metric. All code relating to these can be found in Harry's branch of the git repository. Please also note that Harry wrote the first  beta-VAE model for our group, so some code here is a reorganization, modification, and explansion of some code in Harry's branch.


The list at the bottom of this document correlates the items above with the names of the .py/.ipynb files that generated them.


This directory contains files of the following types and uses for this portion of the project:
 - .py files were used to for training (items 1,2,3)
 - .ipynb files were used for testing and generating figures (items 4,5,6,7)
 - .txt files contain raw values for the losses (items 1,2,3)
 - .h5 files contain trained weights (items 1,2,3)
 - .png files are the saved image results (items 4,5,6,7)
 - .npy files are saved numpy arrays which represent for this project images and quantitative scores (items 5,6,7)


This directory contains the following subdirectories and files (certain naming conventions explained):
Subdirectories:
 - Documents -> contains original Higgins beta VAE paper
 - ds_qual_figs -> contains the figures used in the report and appendix
    - dis_heart_1...  refers to the disentanglement analysis (the latent traversal) for hearts with beta=1
    - dis_sq ... is for latent traversals of squares
    - dis_ov...  is latent traversals of ovals
    - dis_..._1.png or any number .png refers to the intial round of training with 100 epochs (results from item 1)
    - dis_..._1_r2.png or any number_r2.png is for the additional 200 epochs (results from item 2)
    - dis_metric... are the quant score plots (item 7)
    - gt_... are the sample/ground truth images used to test disentanglement and reconstruction after training models
    - loss1.png is the raw loss curve for beta=1 first 100 epochs
    - loss1_r2.png is the raw loss curve for beta=1 for all 300 rounds of training
    - recon1.png are the example images reconstructed for beta=1 after 100 epochs
    - recon1_r2.png are example images reconstructed for beta=1 after 300 epochs

 - round X train... ->  subdirectories are the weights and losses after training (items 1,2,3)
 - tex figs -> select latent traversals and loss curves enhanced for the report

Files:
 - bvae.py -> dependency defining beta vae model and training criterion for sub_tr_134_vae_.py and run_dsprites_models.py
 - dsprites.py -> dependency to work with dsprites dataset for sub_tr_134_vae.py and run_dsprites_models.py
 - dsprites_analysis.ipynb -> notebook for generating loss curves, latent traversals, and qualitative comarisons for first 100 epochs (item 4 and 5 results for item 1)
 - dsprites_analysis_bar_charts.ipynb -> notebook to produce bar charts of quantitative results (item 7)
 - dsprites_analysis_sup_clean.ipynb -> notebook for generating loss curves, latenet traversals, and reconstruction comparison for 2nd round trianing (item 4 and 5 results for item 2)
 - dsprites_ndarray_c... -> this is the dsprites dataset from https://github.com/deepmind/dsprites-dataset (dependency for dsprites.py)
 - README.txt -> this file
 - run_dsprites_models.py -> main file for initial training (item 1)
 - sub_tr_134_vae.py -> main file for supplemental training (items 2 and 3)
 - test_imgs.npy -> sample images used for qualitative analysis; saved for consistency 
 - tf2.1.0_gpu_env_package-list.txt -> lists all packages and versions used in this project (not all packages on the list were used, but all required packages are included on the list)
 - vae5_repeat_accuracies.npy -> raw values of quantitative metric results for the 5 replicates at 100 epochs done for beta=5 (item 7 results from item 3)
 - vae13457_repeat_accuracies.npy -> raw values of quantitative metric results for the additional 200 epochs for beta=1,3,4,5,7 (item 7 results from item2)


Note: If using anaconda, my environment can be replicated using the following command:
>> conda create -n <env-name> --file tf2.1.0_gpu_env_package-list.txt
	(source: conda documentation: https://docs.conda.io/projects/conda/en/latest/commands/list.html)

Please let me know if you have any questions or concerns regarding these files. Thank you for your time and consideration.
