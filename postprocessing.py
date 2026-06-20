# %% [markdown]
# # Postprocessing

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import os
import myfunctions
import h5py
import TM26_unet as unet


GPU = True
MULTIGPU = True

DELETE_OLD_FILES = False
DELETE_OLD_FILES = False

SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
GPU = True

diffLimDir = "dataInputs/"
testDataDir = "output_prep/"
outputDir = "output_post/"
inputDir = "output_infer/"

datasets_name = ["lightfield", "WFSxy", "WFSxy_DM", "WFSxy_DM_intensity", "WFSxy_intensity"]
# datasets_name = ["lightfield", "WFSxy"]
dataset_name_readable = ["lightfield", "WFSxy", "WFSxy and DM commands", "WFSxy, SHWFS intensity map and DM commands", "WFSxy and SHWFS intensity map"]


D = 17.5e-3  # [m]
wl = 632.8e-9  # [m]
pad = 2
dx = 3.928571e-06

SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
GPU = True

for i in range(len(datasets_name)):
    os.makedirs(outputDir + datasets_name[i], exist_ok=True)

dataset_numbers = len(datasets_name)

# %% [markdown]
# ### Inference data

# %%
for dataset_number in range(dataset_numbers):
    print("-------------------------------------------")
    print(f"Starting inference for dataset {dataset_name_readable[dataset_number]} ({dataset_number+1}/{dataset_numbers})")
    print(f"Loading data for dataset {dataset_name_readable[dataset_number]}...")
    if datasets_name[dataset_number] == "lightfield":
        
        with h5py.File(inputDir+datasets_name[dataset_number]+"/predictions.h5", 'r') as temp:
            inferences = temp[list(temp.keys())[0]][()]
        with h5py.File("dataInputs/ipsf_difflim.h5", 'r') as temp:
            dx_scaling = temp[list(temp.keys())[0]][()].shape[1]/inferences.shape[1]
            ipsf_difflim = myfunctions.pixel_adder(temp[list(temp.keys())[0]][()],final_shape=[inferences.shape[1], inferences.shape[2]])   
            
        truths = np.zeros_like(inferences)
        with h5py.File(testDataDir+"ao_ipsf_test_1.h5", 'r') as temp:
            key = list(temp.keys())
            tempdata = temp[key[0]][()][:]
            for i in range(tempdata.shape[0]):
                truths[i,:,:] = myfunctions.pixel_adder(tempdata[i],final_shape=[inferences.shape[1], inferences.shape[2]])
        
    else:
        with h5py.File(inputDir+datasets_name[dataset_number]+"/predictions.h5", 'r') as temp:
            inferences = temp[list(temp.keys())[0]][()]
        with h5py.File("dataInputs/ipsf_difflim.h5", 'r') as temp:
            ipsf_difflim = temp[list(temp.keys())[0]][()]
            dx_scaling = temp[list(temp.keys())[0]][()].shape[1]/inferences.shape[1]
        with h5py.File(testDataDir+"ao_ipsf_test_1.h5", 'r') as temp:
            key = list(temp.keys())
            truths = temp[key[0]][()]
    
    print("Padding data, computing OTFs and Strehl ratios...")

    dx_scaled = dx*dx_scaling
    
    padded_ipsf_difflim = myfunctions.pad_center(ipsf_difflim, pad)        
    padded_truth = myfunctions.pad_center(truths, pad)
    padded_inferences = myfunctions.pad_center(inferences,pad)

    N = ipsf_difflim.shape[1]*pad
    fc = D / wl
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dx_scaled))
    freq_n = np.abs(freq) / fc
    NU_X, NU_Y = np.meshgrid(freq_n, freq_n)
    nu_n_grid = np.sqrt(NU_X**2 + NU_Y**2)
    # FREQ_MASK = np.abs(freq) < fc
    FREQ_MASK = nu_n_grid < 1

    OTF_ipsf_difflim = np.abs(myfunctions.otf_from_psf(padded_ipsf_difflim))
    OTF_tel = myfunctions.telescope_otf(nu_n_grid)
    OTF_aliasing = OTF_ipsf_difflim - OTF_tel
    OTF_truths = (np.abs(myfunctions.otf_from_psf(padded_truth))[...,:,:] - OTF_aliasing)*FREQ_MASK
    OTF_inferences = (np.abs(myfunctions.otf_from_psf(padded_inferences))[...,:,:] - OTF_aliasing)*FREQ_MASK

    inference_Strehls = np.sum(OTF_inferences, axis=(-2, -1)) / np.sum(OTF_tel*FREQ_MASK)
    truths_Strehls = np.sum(OTF_truths, axis=(-2, -1)) / np.sum(OTF_tel*FREQ_MASK)
    Strehls_difference = truths_Strehls - inference_Strehls
    Strehls_rel_fidd = Strehls_difference/truths_Strehls

    therandom = np.random.randint(0, inferences.shape[0])

    plt.figure(figsize=(10, 8))
    plt.plot(freq, OTF_truths[therandom,N//2, :], label="Measured OTF")
    plt.plot(freq, OTF_inferences[therandom,N//2, :]*FREQ_MASK[N//2,:], label="Inference OTF")
    plt.plot(freq, OTF_tel[N//2, :], "k", label="Theoretical diffraction-limited OTF")
    plt.plot(freq, OTF_ipsf_difflim[N//2, :], label="Diffraction-limited OTF\nfrom simulated PSF with aliasing")
    plt.plot(freq, OTF_aliasing[N//2, :],label = "Aliasing OTF")
    plt.plot(freq, OTF_ipsf_difflim[N//2, :]-OTF_aliasing[N//2, :], "--", label="Diffraction-limited OTF\nfrom simulated PSF without aliasing")
    plt.vlines([-fc, fc], 0, 1, colors="r", linestyles="--", label = r"$\pm$ cutoff frequency")
    plt.title(f"Diffraction limited and randomly\nselected OTFs (N°{therandom})", fontsize = BIGGER_SIZE)
    plt.xlabel("Angular frequency [cycles/meter]", fontsize=SMALL_SIZE)
    plt.ylabel("OTF", fontsize=SMALL_SIZE)
    plt.xlim([-fc * 1.2, fc * 1.2])
    plt.ylim([0, 1])
    plt.xticks(fontsize=SMALL_SIZE*.8)
    plt.yticks(fontsize=SMALL_SIZE*.8)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=SMALL_SIZE)
    plt.tight_layout()

    plt.savefig(outputDir + datasets_name[dataset_number]+ f"/{datasets_name[dataset_number]}_iOTF.png")

    lower_bounds = truths_Strehls * 0.9  # -10%
    upper_bounds = truths_Strehls * 1.1  # +10%

    plt.close('all')
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

    # Top plot: Strehl ratios
    ax.plot(truths_Strehls, '.k', label=r'$S_{ref}$', markersize=1)
    ax.plot(inference_Strehls, '.r', label=r'$S_{R}$', markersize=1)
    ax.set_ylim([0, 1])
    ax.fill_between(range(len(truths_Strehls)), lower_bounds, upper_bounds, 
                    color='gray', alpha=0.3, edgecolor='none', label='±10% of expected Strehl ratio')
    ax.set_ylabel('Strehl ratio [-]', fontsize=MEDIUM_SIZE)
    ax.tick_params(axis='y', labelsize=SMALL_SIZE)
    ax.legend(fontsize=SMALL_SIZE)
    ax.set_title("Comparison of Strehl ratios for instantaneous PSFs", fontsize=BIGGER_SIZE)
    ax.set_xlim([0, inference_Strehls.shape[0]])
    ax.grid()

    # Bottom plot: Relative distance (ax2 is now a separate subplot below)
    ax2.plot(Strehls_rel_fidd * 100, '.-k', label=r"$\frac{S_{R}-S_{ref}}{S_{ref}}$",markersize=3)
    ax2.set_ylabel('Relative distance [%]', fontsize=MEDIUM_SIZE, color='k')
    ax2.tick_params(axis='y', labelsize=SMALL_SIZE, colors='k')
    symetric_lim = max(abs(np.min(Strehls_rel_fidd) * 110), abs(np.max(Strehls_rel_fidd) * 110))
    ax2.set_ylim([-symetric_lim, symetric_lim])
    ax2.legend(fontsize=MEDIUM_SIZE)
    ax2.set_xlabel('Iteration [-]', fontsize=MEDIUM_SIZE)
    ax2.set_xlim([0, inference_Strehls.shape[0]])
    ax2.grid()
    ax2.tick_params(axis='x', labelsize=SMALL_SIZE)

    plt.tight_layout()

    plt.savefig(outputDir + datasets_name[dataset_number]+ f"/{datasets_name[dataset_number]}_iStrehl.png")

    with h5py.File(outputDir + datasets_name[dataset_number] + f"/{datasets_name[dataset_number]}_istrehls.h5", 'w') as f:
            f.create_dataset("Measured", data=truths_Strehls)
            f.create_dataset("Inferred", data=inference_Strehls)
            f.create_dataset("Relative", data=Strehls_rel_fidd)

    print(f"Data for dataset {dataset_name_readable[dataset_number]} processed and saved in {outputDir + datasets_name[dataset_number]}")


# %% [markdown]
# ## Rotational average
# 
# 1. Find center
# 2. Average the PSF
# 3. compare


