
"""Statistics-only PSF/OTF processing with tqdm progress reporting.
 
PSF inputs follow the original PSF postprocessing convention: the padded PSF is
converted to an absolute real Fourier array, corrected using a reference-derived
aliasing term, restricted to the telescope cutoff, and integrated for Strehl.
EE and FWHM are computed on the PSF reconstructed from that dealiased,
cutoff-masked OTF.
 
OTF inputs: the Strehl numerator is the volume under the stored OTF after the
treatment chosen with --otf-correction (default 'alias_mask': subtract the
reference aliasing term and apply the telescope cutoff mask, as for PSF inputs
and as defined in the report).  FWHM and EE are computed on the PSF
reconstructed from the *uncorrected* stored OTF, mirroring the PSF branch,
where they come from the uncorrected PSF.  The OTF frequency grid is built from
the actual OTF dimensions.
 
EE conventions (identical for PSF and OTF inputs):
  ee_airy_fraction : fraction of the total PSF energy inside the 1st Airy radius
  ee_times_strehl  : ee_airy_fraction * Strehl
"""
import argparse
import csv
import json
from pathlib import Path
 
import h5py
import numpy as np
from tqdm.auto import tqdm
 
import myfunctions as mf
 
 
PSF_DATASETS = [
    'lightfield', 'lightfield_DM', 'WFSxy', 'WFSxy_DM',
    'WFSxy_DM_intensity', 'WFSxy_intensity',
]
OTF_DATASETS = ['WFSxy_DM_intensity_otf']
 
 
def first_dataset(f):
    names = []
    f.visititems(
        lambda name, obj: names.append(name)
        if isinstance(obj, h5py.Dataset) else None
    )
    if len(names) != 1:
        raise ValueError(
            f'{f.filename}: expected exactly one dataset, found {names}'
        )
    return f[names[0]]
 
 
def image(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f'Expected square 2D image; got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError('Input contains NaN or infinity')
    return a
 
 
def resize(a, shape):
    return a if a.shape == shape else mf.pixel_adder(
        a, final_shape=list(shape)
    )
 
 
def divide(a, b, eps):
    return a / b if np.isfinite(b) and abs(b) > eps else np.nan
 
 
def describe(values, ddof):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if not n:
        return dict(
            n=0, mean=np.nan, std=np.nan, median=np.nan,
            min=np.nan, max=np.nan, p05=np.nan, p95=np.nan,
        )
    return dict(
        n=int(n),
        mean=float(a.mean()),
        std=float(a.std(ddof=ddof)) if n > ddof else np.nan,
        median=float(np.median(a)),
        min=float(a.min()),
        max=float(a.max()),
        p05=float(np.percentile(a, 5)),
        p95=float(np.percentile(a, 95)),
    )
 
 
def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
 
 
def frequency_context(n, spacing, args):
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=spacing))
    xx, yy = np.meshgrid(freq, freq)
    nu = np.hypot(xx, yy) / (args.diameter / args.wavelength)
    tel = mf.telescope_otf(nu)
    mask = nu < 1
    return freq, tel, mask
 
 
def context(ref, shape, kind, args):
    native_n = ref.shape[0]
 
    if kind == 'psf':
        ref = resize(ref, shape)
        spacing = args.dx * native_n / shape[0]
        ref_padded = mf.pad_center(ref, args.pad)
        n = ref_padded.shape[0]
        freq, tel, mask = frequency_context(n, spacing, args)
        alias = np.abs(mf.otf_from_psf(ref_padded)) - tel
        y, x = np.indices(ref_padded.shape)
        airy = (
            np.hypot(x - n // 2, y - n // 2)
            < 1.22 * args.wavelength / args.diameter / spacing
        )
        return dict(
            ref=ref_padded,
            ref_sum=ref_padded.sum(),
            dx=spacing,
            tel=tel,
            mask=mask,
            alias=alias,
            airy=airy,
        )
 
    # OTF mode: the stored OTFs determine the evaluated frequency grid.
    spacing = args.dx
    ref_padded = mf.pad_center(ref, args.pad)
    n = shape[0]
    if ref_padded.shape[0] < n:
        raise ValueError(
            'Padded reference PSF is smaller than the supplied OTF; '
            'increase --pad or supply an OTF consistent with the reference.'
        )
    if ref_padded.shape[0] != n:
        raise ValueError(
            f'Padded reference PSF is {ref_padded.shape[0]} px but the OTF '
            f'is {n} px; the aliasing term cannot be subtracted. Adjust '
            '--pad or supply an OTF consistent with the reference.'
        )
    freq, tel, mask = frequency_context(n, spacing, args)
    alias = np.abs(mf.otf_from_psf(ref_padded)) - tel
    # The aliasing term should vanish at DC (both OTFs are 1 there). If not,
    # the reference OTF is not normalised like the telescope OTF.
    alias_dc = float(alias[n // 2, n // 2])
    if abs(alias_dc) > 1e-3:
        tqdm.write(
            f'WARNING: aliasing term at DC is {alias_dc:.3g} (expected ~0); '
            'the reference OTF is not normalised like the telescope OTF, '
            'so Strehl values with --otf-correction alias_mask are biased.'
        )
    y, x = np.indices((n, n))
    airy = (
        np.hypot(x - n // 2, y - n // 2)
        < 1.22 * args.wavelength / args.diameter / spacing
    )
    return dict(
        ref=ref_padded,
        ref_sum=ref_padded.sum(),
        dx=spacing,
        tel=tel,
        mask=mask,
        alias=alias,
        airy=airy,
    )
 
 
def metrics(a, kind, c, args):
    if kind == 'psf':
        psf = mf.pad_center(a, args.pad)
        if abs(psf.sum()) <= args.epsilon:
            raise ValueError('PSF has zero or near-zero total flux')
        otf = (
            np.abs(mf.otf_from_psf(psf)) - c['alias']
        ) * c['mask']
        strehl = divide(
            otf.sum(),
            (c['tel'] * c['mask']).sum(),
            args.epsilon,
        )
        # EE and FWHM use the PSF reconstructed from the dealiased,
        # cutoff-masked OTF, matching the OTF-input convention.
        psf_metrics = mf.otf_to_psf(otf, c['ref_sum'])
        ee = divide(
            psf_metrics[c['airy']].sum(),
            psf_metrics.sum(),
            args.epsilon,
        )
    else:
        # Strehl: volume under the corrected OTF / volume under the
        # diffraction-limited OTF inside the cutoff (aliasing removed first).
        if args.otf_correction == 'alias_mask':
            otf = (a - c['alias']) * c['mask']
        elif args.otf_correction == 'mask':
            otf = a * c['mask']
        else:  # 'none'
            otf = a
 
        strehl = divide(
            otf.sum(),
            (c['tel'] * c['mask']).sum(),
            args.epsilon,
        )
        # PSF metrics (FWHM, EE) come from the uncorrected stored OTF,
        # as in the PSF branch where they come from the uncorrected PSF.
        psf_metrics = mf.otf_to_psf(a, c['ref_sum'])
        ee = divide(
            psf_metrics[c['airy']].sum(),
            psf_metrics.sum(),
            args.epsilon,
        )
 
    psf_for_geometry = psf_metrics
    fx, fy = mf.fwhm_2d(psf_for_geometry)
    rp = mf.radial_average(psf_for_geometry)
    idx = np.flatnonzero(rp >= rp.max() / 2)
    fr = (idx[-1] - idx[0]) * 2 if idx.size else np.nan
 
    return dict(
        strehl=strehl,
        fwhm_x_rad=fx * c['dx'],
        fwhm_y_rad=fy * c['dx'],
        fwhm_radial_rad=fr * c['dx'],
        ee_airy_fraction=ee,
        ee_times_strehl=ee * strehl,
        negative_pixel_fraction=float(np.mean(psf_for_geometry < 0)),
    )
 
 
def main(kind='psf', long=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=PSF_DATASETS if kind == 'psf' else OTF_DATASETS,
    )
    parser.add_argument('--input-dir', type=Path, default=Path('output_infer'))
    parser.add_argument('--test-dir', type=Path, default=Path('output_prep'))
    parser.add_argument('--model-dir', type=Path, default=Path('output_ML'))
    parser.add_argument('--output-dir', type=Path, default=Path('output_stats'))
    parser.add_argument(
        '--reference',
        type=Path,
        default=Path('dataInputs/ipsf_difflim.h5'),
    )
    parser.add_argument(
        '--exposures',
        type=int,
        nargs='+',
        default=[10, 100, 1000, 5000],
    )
    parser.add_argument('--ddof', type=int, choices=[0, 1], default=0)
    parser.add_argument('--dx', type=float, default=3.928571e-6)
    parser.add_argument('--diameter', type=float, default=17.5e-3)
    parser.add_argument('--wavelength', type=float, default=632.8e-9)
    parser.add_argument('--pad', type=int, default=2)
    parser.add_argument('--epsilon', type=float, default=1e-12)
    parser.add_argument('--otf-truth-scale', type=float, default=1.0)
    parser.add_argument('--otf-prediction-scale', type=float, default=1.0)
    parser.add_argument(
        '--otf-correction',
        choices=['alias_mask', 'mask', 'none'],
        default='alias_mask',
        help="Treatment of stored OTFs before the Strehl sum: subtract the "
             "reference aliasing term and apply the cutoff mask "
             "(alias_mask, default), cutoff mask only (mask), or none.",
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable tqdm progress bars',
    )
    args = parser.parse_args()
    args.long = long
 
    if min(
        args.dx, args.diameter, args.wavelength, args.epsilon
    ) <= 0 or args.pad < 1:
        parser.error(
            'Optical parameters and epsilon must be positive; '
            'pad must be >= 1'
        )
 
    exposures = sorted(set(args.exposures)) if long else [1]
    if not exposures or min(exposures) < 1:
        parser.error('Exposures must be positive integers')
 
    with h5py.File(args.reference, 'r') as f:
        ref = image(first_dataset(f)[()])
 
    truth_path = args.test_dir / (
        'ao_ipsf_test_1.h5' if kind == 'psf' else 'ao_iOTF_test_1.h5'
    )
 
    for name in tqdm(
        args.datasets,
        desc='Datasets',
        unit='dataset',
        position=0,
        dynamic_ncols=True,
        disable=args.no_progress,
    ):
        tqdm.write(f'Loading {name}...')
        folder = args.output_dir / (
            name + ('_long' if long else '_instant')
        )
        folder.mkdir(parents=True, exist_ok=True)
        records, summary, skipped = [], [], []
 
        with h5py.File(
            args.input_dir / name / 'predictions.h5', 'r'
        ) as prediction_file, h5py.File(truth_path, 'r') as truth_file:
            pred = first_dataset(prediction_file)
            truth = first_dataset(truth_file)
 
            if len(pred) != len(truth) or len(pred) == 0:
                raise ValueError(
                    f'{name}: unequal or empty sample counts'
                )
 
            shape = image(pred[0]).shape
            if kind == 'otf' and image(truth[0]).shape != shape:
                raise ValueError(
                    'OTF truth and prediction shapes must match; '
                    'no OTF resampling is performed'
                )
 
            c = context(ref, shape, kind, args)
 
            if kind == 'psf':
                values = np.genfromtxt(
                    args.model_dir / name / 'max_val.txt',
                    delimiter=',',
                    dtype=None,
                    encoding='utf-8',
                )
                truth_scale = 1.0 / float(values[1][1])
                pred_scale = 1.0
            else:
                truth_scale = args.otf_truth_scale
                pred_scale = args.otf_prediction_scale
 
            warned_strehl = False
            for exposure in exposures:
                n_windows = len(pred) // exposure
                if n_windows == 0:
                    skipped.append(
                        dict(
                            exposure=exposure,
                            reason='insufficient frames',
                        )
                    )
                    tqdm.write(
                        f'{name}: skipping exposure={exposure}; '
                        f'only {len(pred)} frames available'
                    )
                    continue
 
                group = []
                with tqdm(
                    total=n_windows * exposure,
                    desc=(
                        f'{name} | {exposure} frame(s)/exposure'
                    ),
                    unit='frame',
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                    disable=args.no_progress,
                ) as progress:
                    for window in range(n_windows):
                        progress.set_postfix(
                            window=f'{window + 1}/{n_windows}',
                            refresh=False,
                        )
                        start = window * exposure
                        stop = (window + 1) * exposure
 
                        prediction_average = np.zeros(shape)
                        truth_average = np.zeros(shape)
 
                        for index in range(start, stop):
                            prediction_average += (
                                image(pred[index]) * pred_scale
                            )
                            truth_average += (
                                resize(image(truth[index]), shape)
                                * truth_scale
                            )
                            if index < stop - 1:
                                progress.update(1)
 
                        prediction_average /= exposure
                        truth_average /= exposure
 
                        truth_metrics = metrics(
                            truth_average, kind, c, args
                        )
                        if (
                            np.isfinite(truth_metrics['strehl'])
                            and truth_metrics['strehl'] > 1.05
                            and not warned_strehl
                        ):
                            warned_strehl = True
                            tqdm.write(
                                f'WARNING: {name}: truth Strehl '
                                f'{truth_metrics["strehl"]:.3f} > 1; check '
                                'aliasing removal / OTF scaling '
                                '(--otf-correction, --otf-truth-scale).'
                            )
                        prediction_metrics = metrics(
                            prediction_average, kind, c, args
                        )
 
                        row = dict(
                            dataset=name,
                            exposure_frames=exposure,
                            window=window,
                            start_frame=start,
                            stop_frame_exclusive=stop,
                        )
 
                        for metric in truth_metrics:
                            truth_value = truth_metrics[metric]
                            prediction_value = prediction_metrics[metric]
                            error = prediction_value - truth_value
                            row.update(
                                {
                                    metric + '_truth': truth_value,
                                    metric + '_prediction': prediction_value,
                                    metric + '_error': error,
                                    metric + '_relative_error_pct':
                                        100 * divide(
                                            error,
                                            truth_value,
                                            args.epsilon,
                                        ),
                                }
                            )
 
                        records.append(row)
                        group.append(row)
                        progress.update(1)
 
                for metric in truth_metrics:
                    for series in (
                        'truth', 'prediction', 'error', 'relative_error_pct'
                    ):
                        key = metric + '_' + series
                        values = np.array(
                            [item[key] for item in group]
                        )
                        statistics = describe(values, args.ddof)
                        finite = values[np.isfinite(values)]
                        entry = dict(
                            dataset=name,
                            exposure_frames=exposure,
                            metric=metric,
                            series=series,
                            ddof=args.ddof,
                            total_windows=n_windows,
                            nonfinite_count=int(
                                len(values) - len(finite)
                            ),
                            discarded_frames=len(pred) % exposure,
                            **statistics,
                        )
                        if series in (
                            'error', 'relative_error_pct'
                        ):
                            entry.update(
                                mae=(
                                    float(np.mean(np.abs(finite)))
                                    if finite.size else np.nan
                                ),
                                rmse=(
                                    float(
                                        np.sqrt(
                                            np.mean(finite ** 2)
                                        )
                                    )
                                    if finite.size else np.nan
                                ),
                            )
                        summary.append(entry)
 
                tqdm.write(
                    f'{name}: exposure={exposure}, '
                    f'windows={n_windows}, '
                    f'discarded={len(pred) % exposure}'
                )
 
        tqdm.write(f'Writing CSV files for {name}...')
        write_csv(folder / 'per_sample_metrics.csv', records)
        write_csv(folder / 'summary_statistics.csv', summary)
 
        metadata = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        metadata.update(
            metric_version='otf_alias_dealiased_psf_v6',
            strehl_definition=(
                'sum(corrected OTF) / '
                'sum(theoretical telescope OTF inside cutoff)'
            ),
            psf_otf_processing=(
                'abs(real FFT of flux-normalized padded PSF), '
                'minus reference alias term, then cutoff mask'
            ),
            otf_input_processing=(
                'Strehl numerator uses the stored OTF treated according to '
                'otf_correction (alias_mask: minus reference alias term, '
                'then cutoff mask); FWHM/EE use the uncorrected OTF'
            ),
            psf_input_processing=(
                'Strehl uses the OTF from the padded PSF after reference-alias '
                'subtraction and cutoff masking; FWHM/EE use the PSF '
                'reconstructed from that dealiased, cutoff-masked OTF'
            ),
            otf_frequency_grid=(
                'constructed from the actual stored OTF dimensions'
            ),
            psf_metrics=(
                'real PSF reconstructed from the dealiased, cutoff-masked '
                'OTF in both PSF and OTF modes'
            ),
            ee_definition=(
                'ee_airy_fraction: fraction of energy of the PSF '
                'reconstructed from the dealiased, cutoff-masked OTF within '
                'the first Airy radius (same for PSF and OTF); '
                'ee_times_strehl = ee_airy_fraction * strehl'
            ),
            recenter_before_averaging=False,
            clipping=False,
            effective_exposures=exposures,
            kind=kind,
            error_sign='prediction - truth',
            windows=(
                'non-overlapping; incomplete final window discarded'
            ),
            skipped=skipped,
        )
        (folder / 'settings.json').write_text(
            json.dumps(metadata, indent=2)
        )
        tqdm.write(f'Saved statistics: {folder}')
 