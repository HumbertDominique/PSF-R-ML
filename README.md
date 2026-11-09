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
|postprocessing.ipynb|Jupyter notebook used to develop the other postprocessing scripts.|||-|
|postprocessing.py|Instantaneous data postprocessing script for models outputing a PSF|inferences, normalisation values, diffraction limited iPSF|iFWMH, a random iOTF, iEE, iStrehl|-|
|postprocessing_long.py|Long exposure postprocessing script for models outputing a PSF|inferences, normalisation values, diffraction limited iPSF|FWMH, OTFs, EE, Strehl|-|
|postprocessing_otf.py|Instantaneous data postprocessing script for models outputing an OTF|inferences, normalisation values, diffraction limited iPSF|iFWMH, a random iOTF, iEE, iStrehl|-|
|postprocessing_otf_long.py|Long exposure postprocessing script for models outputing an OTF|inferences, normalisation values, diffraction limited iPSF|FWMH, OTFs, EE, Strehl|-|
|myfunctions.py|All helper functions, either new or reused from PA 2025|-|-|-|
|magnitude_swipe_processing.ipynb|Processing parameter swiped datasets to output the corresponding RWFE plots of the AO regulation and the Strehl ratios plot of the long exposures PSF|||-|
|misc.ipynb|Jupyter notebook used for computations taht are not worht a complete script|synthetic iPSFs|iOTFs for training, validatiaon and test|---|
|---|---|---|---|---|


## Statistics-only postprocessing

Four additional scripts compute numerical statistics instead of generating plots. These are rewritten alternatives to the original postprocessing scripts, not modifications of them. The original scripts remain unchanged.

### Files

| File | Purpose |
|---|---|
| `postprocessing_stats.py` | Statistics for instantaneous PSF predictions |
| `postprocessing_long_stats.py` | Statistics for long-exposure PSF predictions |
| `postprocessing_otf_stats.py` | Statistics for instantaneous OTF predictions |
| `postprocessing_otf_long_stats.py` | Statistics for long-exposure OTF predictions |
| `postprocessing_statistics_common.py` | Shared data loading, metric computation, and statistical analysis |

Place all five files in the repository root, beside `myfunctions.py`. They use the existing project dependencies, including NumPy, h5py, and pandas. No GPU is required.

### Usage

Run the scripts from the repository root:

    python postprocessing_stats.py
    python postprocessing_long_stats.py
    python postprocessing_otf_stats.py
    python postprocessing_otf_long_stats.py

To select a dataset and exposure lengths:

    python postprocessing_long_stats.py --datasets WFSxy_DM_intensity --exposures 10 100 1000 5000

Only select datasets whose inference files are available. Use `--help` to list options for input/output paths, optical parameters, normalization scales, and standard-deviation settings.

### Metrics and statistics

The scripts compute:

- Strehl ratio.
- FWHM along x and y, and from the radial profile, in radians.
- Encircled-energy fraction within the centered first Airy disk.
- Encircled-energy fraction multiplied by Strehl, exported separately for comparison with the original OTF scripts.
- Fraction of negative pixels in the evaluated PSF.

For each metric, statistics are computed separately for the ground truth, predictions, signed errors, and relative errors.

The error convention is consistent across metrics:

    error = prediction - ground_truth
    relative_error_pct = 100 * (prediction - ground_truth) / ground_truth

Summary statistics include count, mean, standard deviation, median, minimum, maximum, and the 5th and 95th percentiles. Error summaries also include mean absolute error (MAE) and root mean square error (RMSE).

To examine reconstruction-error scatter, select `series = error` in the summary and read the `std` column. The corresponding `mean` reports the signed bias.

### Standard deviation

By default, the scripts calculate sample standard deviation with `ddof=1`. Use `--ddof 0` to calculate population standard deviation instead.

Statistics are calculated separately for each dataset, metric, series, and exposure duration. Standard deviation is not calculated across different exposure durations.

With `ddof=1`, a single observation produces `NaN` for standard deviation rather than zero. Relative errors with a near-zero ground-truth denominator also produce `NaN`; finite-value counts and excluded-value counts are recorded.

The standard deviation of relative errors is expressed in percentage points.

### Long-exposure processing

For each requested exposure length of N frames, the long-exposure scripts:

1. Divide the sequence into complete, non-overlapping windows of N frames.
2. Average the PSF or OTF frames within each window.
3. Compute the metrics for each averaged window.
4. Calculate statistics across windows of the same duration.

Incomplete final windows are discarded, and the discarded frame count is reported. Exposure lengths exceeding the available frame count are skipped and recorded in `settings.json`.

A single complete window provides a metric value but cannot provide a sample standard deviation. Reported standard deviations describe observed scatter; they are not standard errors or confidence intervals. Non-overlapping windows may still be temporally correlated.

### Inputs and outputs

Default inputs follow the existing project structure:

| Input | Default location |
|---|---|
| Predictions | `output_infer/<dataset>/predictions.h5` |
| PSF ground truth | `output_prep/ao_ipsf_test_1.h5` |
| OTF ground truth | `output_prep/ao_iOTF_test_1.h5` |
| Diffraction-limited PSF | `dataInputs/ipsf_difflim.h5` |
| PSF normalization values | `output_ML/<dataset>/max_val.txt` |

Outputs are saved under `output_stats/<dataset>_instant/` or `output_stats/<dataset>_long/`.

| Output | Contents |
|---|---|
| `per_sample_metrics.csv` | Metric values and paired errors for each frame or exposure window |
| `summary_statistics.csv` | Aggregated statistics, including standard deviation |
| `settings.json` | Processing parameters, error convention, and skipped exposure lengths |

### Differences from the original scripts

These scripts are not guaranteed to reproduce every original output numerically:

- Long-exposure OTF processing averages the frames in each window. The original `postprocessing_otf_long.py` selects an indexed frame after slicing the exposure.
- Ordinary encircled-energy fraction is exported as `ee_airy_fraction`. The additional multiplication by Strehl used in the original OTF scripts is exported separately as `ee_times_strehl`.
- Errors consistently use `prediction - ground_truth`; some original plots use the opposite sign.
- PSF processing retains the original alias-correction conventions and pixel-based FWHM estimators. Instantaneous and long-exposure PSF processing retain their original differences in the PSF used for FWHM and encircled-energy evaluation.
- Negative PSF values are not clipped, and the Airy mask is centered on the array rather than recentered on each peak.

PSF predictions are assumed to be normalized already; PSF ground truth is divided by the saved training maximum. OTF arrays are used as stored by default. Use `--otf-truth-scale` and `--otf-prediction-scale` if different scaling is required.

See `README_statistics.md` for additional implementation details and assumptions.

### Validation status

The generated scripts passed Python syntax checks and in-memory checks for standard deviation, single-observation handling, nonfinite-value filtering, zero-denominator handling, and synthetic PSF metrics.

Full HDF5 end-to-end testing and validation against production inference data have not been performed. Test locally and compare results with the original processing pipeline before using the statistics in scientific reporting.