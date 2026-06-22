# PFSr-ML
PSF reconstruction using Machine learning

## Quickstart
1. Install cuda.
2. `Pip install -r requirements.txt`.
3. Install pytorch according to https://pytorch.org/get-started/locally/.
4. Place input data in _dataInputs_ folder.

## sadadasadadad

|file name|function|Data inputs|Data outputs|Note|
|---|---|---|---|---|
|PA_25_unet.py|Original unet model from PA 2025 using keras and tensorflow|-|-|-|
PA25_ML.ipynb|Jupyter notebook used for training _PA\_25\_unet.py_ on data in PA 2025|-|-|-|
|TM26_unet.py|unet model rewriten with pytorch|-|-|-|
|preprocessing.ipynb|processes the raw data into AO performance plot and split datasets|metadata.txt, ao_DM_1.h5, ao_WFS_1.h5, ao_lightfield_1.h5, ao_ipsf_1.h5 |RWFE plot, training, validation and test dataset for all data inputs, split sequence|-|
|unet_26.ipynb|Jupyter Notebook used to developp the training of _unet\_26.py_ on the data |training and validation dataset for all data inputs| loss history, normalisation values for the training and test dataset, hyperparameters and model weights|The outputs will be in output_ML|
|TM26_unet_run.py|Same as unet_26.ipynb but as a python only file in order to run on a remote server. In order to changer the training data, the loading part of the script has to be replpaced by the propper one. They can be found under their own sections in _unet\_26.ipynb_|Same as _unet\_26.ipynb_|Same as _unet\_26.ipynb_|-|
|inference.ipynb|Jupyter notebook used to develop the inference script.|model.weights.pth, history.h5 and the test data for each dataset.|Training and validation loss plot, test loss plot, predictions|-|
|inference.py|inference script that can be run on a remote server|model.weights.pth, history.h5 and the test data for each dataset.|Training and validation loss plot, test loss plot, predictions|-|
|postprocessing.ipynb||||-|
|postprocessing.py||||-|
|postprocessing_long.py||||-|
|myfunctions.py|All helper functions, either new or reused from PA 2025|-|-|-|
|magnitude_swipe_processing.ipynb|Processing parameter swiped datasets to output the corresponding RWFE plots of the AO regulation and the Strehl ratios plot of the long exposures PSF|||-|
