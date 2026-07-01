####
# Postprocessing script for long exposure PSF
# TM 2026, Dominique Humbert
#
####
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import os
import myfunctions
import h5py


SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

diffLimDir = "dataInputs/"
testDataDir = "output_prep/"
outputDir = "output_post/"
inputDir = "output_infer/"

exposures = [10, 100, 1000, 5000]
# exposures = [10, 100]

datasets_name = ["WFSxy_DM_intensity_otf"]

dataset_name_readable = ["WFSxy with DM commands and\nintensity map to OTF"]


D = 17.5e-3  # [m]
wl = 632.8e-9  # [m]
pad = 2
dx = 3.928571e-06

SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
GPU = True

for i in range(len(datasets_name)):
    os.makedirs(outputDir + datasets_name[i]+"_long", exist_ok=True)

dataset_numbers = len(datasets_name)

for dataset_number in range(dataset_numbers):
    print("-------------------------------------------")
    print(f"Starting postprocessing for dataset {dataset_name_readable[dataset_number]} ({dataset_number+1}/{dataset_numbers})")
    print(f"Loading data for dataset {dataset_name_readable[dataset_number]}...")
    
    with h5py.File(inputDir+datasets_name[dataset_number]+"/predictions.h5", 'r') as temp:
            inferences = temp[list(temp.keys())[0]][()]
    with h5py.File("dataInputs/ipsf_difflim.h5", 'r') as temp:
            ipsf_difflim = temp[list(temp.keys())[0]][()]
            dx_scaling = temp[list(temp.keys())[0]][()].shape[1]/inferences.shape[1]
    with h5py.File(testDataDir+"ao_iOTF_test_1.h5", 'r') as temp:
            key = list(temp.keys())
            truths = temp[key[0]][()]
    x_max = np.genfromtxt("output_ML/"+datasets_name[dataset_number]+"/max_val.txt", delimiter=',', dtype=None, encoding='utf-8')[0][1]
    y_max = np.genfromtxt("output_ML/"+datasets_name[dataset_number]+"/max_val.txt", delimiter=',', dtype=None, encoding='utf-8')[1][1]
    print("Padding data, computing OTFs and Strehl ratios...")

    dx_scaled = dx*1

    padded_ipsf_difflim = myfunctions.pad_center(ipsf_difflim, pad)/ipsf_difflim.max()

    OTF_truths = np.zeros((len(exposures),*padded_ipsf_difflim.shape))
    OTF_inferences = np.zeros_like(OTF_truths)
    inference_Strehls = np.zeros(len(exposures))
    truths_Strehls = np.zeros(len(exposures))
    Strehls_difference = np.zeros(len(exposures))
    Strehls_rel_diff = np.zeros(len(exposures))

    otf = np.zeros((len(exposures),*padded_ipsf_difflim.shape))
    psfs_inferred_from_otf = np.zeros_like(otf)
    psfs_truth_from_otf = np.zeros_like(otf)

    for i, exposure in enumerate(exposures):
        print(f"Computing OTF and Strehl ratios for exposure of {exposure} iterations ({i}/{len(exposures)})")

        padded_truth = truths[0:exposure,:,:]
        padded_inferences = inferences[0:exposure,:,:]

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
        OTF_aliasing =OTF_ipsf_difflim - OTF_tel
        # OTF_truths[i] = (padded_truth[i] - OTF_aliasing)*FREQ_MASK
        # OTF_inferences[i] = (padded_inferences[i] - OTF_aliasing)*FREQ_MASK
        OTF_truths[i] = padded_truth[i]
        OTF_inferences[i] = padded_inferences[i]

        inference_Strehls[i] = np.sum(OTF_inferences[i]) / np.sum(OTF_tel*FREQ_MASK)
        truths_Strehls[i] = np.sum(OTF_truths[i]) / np.sum(OTF_tel*FREQ_MASK)
        Strehls_difference[i] = truths_Strehls[i] - inference_Strehls[i]
        Strehls_rel_diff[i] = Strehls_difference[i]/truths_Strehls[i]

        plt.figure(figsize=(10, 8))
        plt.plot(freq, OTF_tel[N//2, :], ".-b", label="Theoretical diffraction-limited OTF")
        plt.plot(freq, OTF_ipsf_difflim[N//2, :], "k", linestyle="dotted", label="Diffraction-limited OTF\nfrom simulated PSF with aliasing")
        plt.plot(freq, OTF_aliasing[N//2, :],"--.",label = "Aliasing OTF")
        plt.plot(freq, OTF_truths[i][N//2, :],"--k", label="Measured OTF without aliasing")
        plt.plot(freq, OTF_inferences[i][N//2, :],'r', label="Inference OTF")
        plt.plot(freq, OTF_ipsf_difflim[N//2, :]-OTF_aliasing[N//2, :], "--.k", label="Diffraction-limited OTF\nfrom simulated PSF without aliasing")
        plt.vlines([-fc, fc], 0, 1, "k", linestyles="dotted", label = r"$\pm$ cutoff frequency")
        plt.title(f"Diffraction limited and long exposure ({exposure}it) OTFs", fontsize = BIGGER_SIZE)
        plt.xlabel("Angular frequency [cycles/meter]", fontsize=SMALL_SIZE)
        plt.ylabel("OTF", fontsize=SMALL_SIZE)
        plt.xlim([-fc * 1.2, fc * 1.2])
        plt.ylim([0, 1])
        plt.xticks(fontsize=SMALL_SIZE*.8)
        plt.yticks(fontsize=SMALL_SIZE*.8)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=SMALL_SIZE)
        plt.tight_layout()
        plt.show()

        plt.savefig(outputDir + datasets_name[dataset_number]+"_long"+f"/{datasets_name[dataset_number]}_OTF{exposure}.png")

        # psfs_inferred_from_otf[i] = ((np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(OTF_inferences[i], axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))).real)**2
        # psfs_truth_from_otf[i] = ((np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(OTF_truths[i], axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))).real)**2
        otf = OTF_inferences#myfunctions.otf_from_psf(padded_inferences)
        psfs_inferred_from_otf[i] = myfunctions.otf_to_psf(otf[i],padded_ipsf_difflim.sum())
        otf_real = OTF_truths#myfunctions.otf_from_psf(padded_truth)
        psfs_truth_from_otf[i] = myfunctions.otf_to_psf(otf_real[i],padded_ipsf_difflim.sum())

    truths_Strehls = np.asarray(truths_Strehls)
    inference_Strehls = np.asarray(inference_Strehls)

    # truth_errors = truths_Strehls * 0.10

    print(f"Strehl ratios for {datasets_name[dataset_number]} dataset:")
    for i, exposure in enumerate(exposures):
        print(f"Exposure: {exposure} iterations, Measured Strehl: {truths_Strehls[i]:.4e}, Inferred Strehl: {inference_Strehls[i]:.4e}")

    x = np.arange(len(exposures))
    truth_errors = truths_Strehls * 0.10

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, truths_Strehls, yerr=truth_errors, fmt='.k', ecolor='k', capsize=3, label = r"True value $\pm$10%", zorder=1)
    plt.plot(x, inference_Strehls, '.r',label = "Infered value", zorder=3)

    plt.title(dataset_name_readable[dataset_number] + f"\nStrehl ratios for an integration time\n of multiple iterations\n", fontsize=BIGGER_SIZE)
    plt.xticks(x, exposures,fontsize=SMALL_SIZE*.8)
    plt.xlim([-0.5,len(exposures)-0.5])
    plt.xlabel(f"Integration time [it]", fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE*.8)
    plt.ylim([0, 1])
    plt.ylabel("Strehl ratio [-]", fontsize=SMALL_SIZE)
    plt.legend(fontsize=SMALL_SIZE)
    plt.grid()
    plt.tight_layout()

    plt.savefig(outputDir + datasets_name[dataset_number]+"_long" f"/{datasets_name[dataset_number]}_Strehl.png")

    inferred_FWHM_x = []
    inferred_FWHM_y = []

    for i in range(psfs_inferred_from_otf.shape[0]):
        FWHM_x, FWHM_y = myfunctions.fwhm_2d(psfs_inferred_from_otf[i])
        inferred_FWHM_x.append(dx_scaled*FWHM_x)
        inferred_FWHM_y.append(dx_scaled*FWHM_y)

    measured_FWHM_x = []
    measured_FWHM_y = []
    for i in range(psfs_truth_from_otf.shape[0]):
        FWHM_x, FWHM_y = myfunctions.fwhm_2d(psfs_truth_from_otf[i])
        measured_FWHM_x.append(dx_scaled*FWHM_x)
        measured_FWHM_y.append(dx_scaled*FWHM_y)

    # radial average
    inference_fwhm_r = []
    for i in range(psfs_inferred_from_otf.shape[0]):
        rp = myfunctions.radial_average(psfs_inferred_from_otf[i])
        half_max = rp.max() / 2
        idx = np.where(rp >= half_max)[0]
        inference_fwhm_r.append(dx_scaled*(idx[-1] - idx[0])*2)
    # print(f"FWHM of the inferred PSF: {inference_fwhm_r[0]} pixels")
    measured_fwhm_r = []
    for i in range(psfs_truth_from_otf.shape[0]):
        rp = myfunctions.radial_average(psfs_truth_from_otf[i])
        half_max = rp.max() / 2
        idx = np.where(rp >= half_max)[0]
        measured_fwhm_r.append(dx_scaled*(idx[-1] - idx[0])*2)
    # print(f"FWHM of the measured PSF: {measured_fwhm_r[0]} pixels")

    diff_FWHM_x = np.array(inferred_FWHM_x) - np.array(measured_FWHM_x)
    diff_FWHM_y = np.array(inferred_FWHM_y) - np.array(measured_FWHM_y)
    diff_FWHM_r = np.array(inference_fwhm_r) - np.array(measured_fwhm_r)
    difflim_FWMH = 1.028 * wl / (D)

    print(f"FWHM of the diffraction-limited PSF: {difflim_FWMH:.2e} pixels")
    print(f"FWHM_r for {datasets_name[dataset_number]} dataset:")
    for i, exposure in enumerate(exposures):
        print(f"Exposure: {exposure} iterations, Measured FWHM_r: {measured_fwhm_r[i]:.4e}, Inferred FWHM_r: {inference_fwhm_r[i]:.4e}")


    x = np.arange(len(exposures))

    fig = plt.figure(figsize=(12, 10)) # https://matplotlib.org/3.5.0/tutorials/intermediate/gridspec.html
    gs = GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[3.2, 1.3],
        height_ratios=[.5, 1, 1],
        wspace=0.25,
        hspace=0.35
    )

    ax_text    = fig.add_subplot(gs[0, 0:])
    ax_avg     = fig.add_subplot(gs[1, 0])
    ax_dy      = fig.add_subplot(gs[2, 0], sharex=ax_avg)
    ax_dx      = fig.add_subplot(gs[1:, 1])

    # --- Top-left: title + explanation ---
    ax_text.axis("off")
    ax_text.text(0.5, 1, f"FWHM computation results", ha="center", va="center", fontsize=BIGGER_SIZE)
    ax_text.text(0.5, 0.25, f"Difference between the inferred and measured FWHM in x, y \nand the radial average for each long exposures from the test set.\nThe annotation next to each point indicates the measured FWHM value in pixels.\nAs reference, the diffraction-limited FWHM is {difflim_FWMH:.2e} pixels.",ha="center", va="center", fontsize=18)
    # --- Top-right: contour plot ---

    # --- bottom plots ---
    x = np.arange(len(exposures))
    # delta_avg, delta_y, delta_x should be your arrays

    ax_avg.plot(x, diff_FWHM_r, '.k')
    ax_avg.set_ylabel(r'$\Delta_{avg}$ [rad]', fontsize=18)
    ax_avg.tick_params(axis='both', labelsize=12)
    # ax_avg.set_yticks(x)
    ax_avg.grid(True, alpha=0.3)
    ylim = max(abs(np.min(diff_FWHM_r)), abs(np.max(diff_FWHM_r)), dx_scaled) * 1.1
    ax_avg.set_ylim([-ylim, ylim])
    for i, val in enumerate(measured_fwhm_r):
        ax_avg.annotate(f"{val:.0f}", (x[i], diff_FWHM_r[i]),
                    textcoords="offset points", xytext=(6, 6),
                    ha='left', va='bottom', fontsize=12)

    ax_dy.plot(x, diff_FWHM_y, '.k')
    ax_dy.set_xticks(x,exposures)
    ax_dy.set_xlabel(f"Integration time [it]", fontsize=18)
    ax_dy.set_ylabel(r'$\Delta_y$ [rad]', fontsize=18)
    ax_dy.tick_params(axis='both', labelsize=12)
    ax_dy.grid(True, alpha=0.3)
    ylim = max(abs(np.min(diff_FWHM_y)), abs(np.max(diff_FWHM_y)), dx_scaled) * 1.1
    ax_dy.set_ylim([-ylim, ylim])
    for i, val in enumerate(measured_FWHM_y):
        ax_dy.annotate(f"{val:.0f}", (x[i], diff_FWHM_y[i]),
                   textcoords="offset points", xytext=(6, 6),
                   ha='left', va='bottom', fontsize=12)

    ax_dx.plot(diff_FWHM_x, x, '.k')
    ax_dx.set_xlabel(r'$\Delta_x$ [rad]', fontsize=18)
    ax_dx.yaxis.set_label_position("right")
    ax_dx.set_yticks(x,exposures)
    ax_dx.yaxis.tick_right()
    ax_dx.set_ylabel(f"Integration time [it]", fontsize=18)
    ax_dx.tick_params(axis='both', labelsize=12)
    ax_dx.grid(True, alpha=0.3)
    xlim = max(abs(np.min(diff_FWHM_x)), abs(np.max(diff_FWHM_x)), dx_scaled) * 1.1
    ax_dx.set_xlim([-xlim, xlim])
    for i, val in enumerate(measured_FWHM_x):
        ax_dx.annotate(f"{val:.0f}", (diff_FWHM_x[i], x[i]),
                   textcoords="offset points", xytext=(6, 0),
                   ha='left', va='center', fontsize=12)

    ax_avg.tick_params(labelbottom=False)

    plt.savefig(outputDir + datasets_name[dataset_number]+"_long" f"/{datasets_name[dataset_number]}_FWHM.png")

    airy_r = 1.22*wl/D
    airy_r_px = airy_r/dx_scaled
    airy_d_px = 2*airy_r_px
    AIRY_MASK = np.zeros_like(psfs_inferred_from_otf[0])

    # Airy disk mask
    y, x = np.indices(AIRY_MASK.shape)
    rr = np.sqrt((x - AIRY_MASK.shape[1] // 2) ** 2 + (y - AIRY_MASK.shape[0] // 2) ** 2)
    AIRY_MASK[rr < airy_r_px] = 1



    norm_psfs_inferred_from_otf = (psfs_inferred_from_otf / np.sum(psfs_inferred_from_otf,axis=(-2, -1), keepdims=True))
    norm_psfs_truth_from_otf = (psfs_truth_from_otf / np.sum(psfs_truth_from_otf,axis=(-2, -1), keepdims=True))

    norm_psfs_inferred_from_otf *= inference_Strehls[:, None, None]
    norm_psfs_truth_from_otf *= truths_Strehls[:, None, None]

    EE_inferred = np.sum(AIRY_MASK*norm_psfs_inferred_from_otf,axis=(-2, -1))
    EE_measured = np.sum(AIRY_MASK*norm_psfs_truth_from_otf,axis=(-2, -1))
    EE_diff = EE_measured - EE_inferred
    EE_rel_diff = EE_diff/EE_measured

    print(f"EE within the 1st Airy disc for {datasets_name[dataset_number]} dataset:")
    for i, exposure in enumerate(exposures):
        print(f"Exposure: {exposure} iterations, Measured EE: {EE_measured[i]:.4e}, Inferred EE: {EE_inferred[i]:.4e}")



    x = np.arange(len(exposures))

    truth_errors = EE_measured * 0.10

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, EE_measured, yerr=truth_errors, fmt='.k', ecolor='k', capsize=3, label = r"True value $\pm$10%", zorder=1)
    plt.plot(x, EE_inferred, '.r', label = "Infered value", zorder=3)

    plt.title(dataset_name_readable[dataset_number] + f"\nEE within the 1st Airy disc for an integration\ntime of multiple iterations\n", fontsize=BIGGER_SIZE)
    plt.xticks(x, exposures,fontsize=SMALL_SIZE*.8)
    plt.xlim([-0.5,len(exposures)-0.5])
    plt.xlabel(f"Integration time [it]", fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE*.8)
    plt.ylim([0, 1])
    plt.ylabel('$EE_{Airy1}$ [-]', fontsize=SMALL_SIZE)
    plt.legend(fontsize=SMALL_SIZE)
    plt.grid()
    plt.tight_layout()

    plt.savefig(outputDir + datasets_name[dataset_number]+"_long" f"/{datasets_name[dataset_number]}_EE.png")

print("Processing completes")