"""Statistics-only PSF/OTF processing with tqdm progress reporting."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from tqdm.auto import tqdm
import myfunctions as mf

PSF_DATASETS = ['lightfield', 'lightfield_DM', 'WFSxy', 'WFSxy_DM',
                'WFSxy_DM_intensity', 'WFSxy_intensity']


def first_dataset(f):
    names = []
    f.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
    if len(names) != 1:
        raise ValueError(f'{f.filename}: expected exactly one dataset, found {names}')
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
    return a if a.shape == shape else mf.pixel_adder(a, final_shape=list(shape))


def divide(a, b, eps):
    return a / b if np.isfinite(b) and abs(b) > eps else np.nan


def describe(values, ddof):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if not n:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan,
                    min=np.nan, max=np.nan, p05=np.nan, p95=np.nan)
    return dict(n=int(n), mean=float(a.mean()),
                std=float(a.std(ddof=ddof)) if n > ddof else np.nan,
                median=float(np.median(a)), min=float(a.min()), max=float(a.max()),
                p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def context(ref, shape, kind, args):
    native_n = ref.shape[0]
    if kind == 'psf':
        ref = resize(ref, shape)
        spacing = args.dx * native_n / shape[0]
        ref = mf.pad_center(ref, args.pad)
    else:
        spacing = args.dx
        ref = mf.pad_center(ref, args.pad)
        if ref.shape != shape:
            raise ValueError(f'OTF shape {shape} does not match padded reference {ref.shape}')
    n = ref.shape[0]
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=spacing))
    xx, yy = np.meshgrid(freq, freq)
    nu = np.hypot(xx, yy) / (args.diameter / args.wavelength)
    tel = mf.telescope_otf(nu)
    mask = nu < 1
    alias = np.abs(mf.otf_from_psf(ref)) - tel
    y, x = np.indices(ref.shape)
    airy = np.hypot(x - n//2, y - n//2) < 1.22*args.wavelength/args.diameter/spacing
    return dict(ref=ref, dx=spacing, tel=tel, mask=mask, alias=alias, airy=airy)


def metrics(a, kind, c, args):
    if kind == 'psf':
        psf = mf.pad_center(a, args.pad)
        if abs(psf.sum()) <= args.epsilon:
            raise ValueError('PSF has zero or near-zero total flux')
        otf = (np.abs(mf.otf_from_psf(psf)) - c['alias']) * c['mask']
    else:
        otf = a
        psf = mf.otf_to_psf(otf, c['ref'].sum())
    strehl = divide(otf.sum(), (c['tel']*c['mask']).sum(), args.epsilon)
    if kind == 'psf' and not args.long:
        psf = mf.otf_to_psf(otf, psf.sum())
    fx, fy = mf.fwhm_2d(psf)
    rp = mf.radial_average(psf)
    idx = np.flatnonzero(rp >= rp.max()/2)
    fr = (idx[-1]-idx[0])*2 if idx.size else np.nan
    ee = divide(psf[c['airy']].sum(), psf.sum(), args.epsilon)
    return dict(strehl=strehl, fwhm_x_rad=fx*c['dx'], fwhm_y_rad=fy*c['dx'],
                fwhm_radial_rad=fr*c['dx'], ee_airy_fraction=ee,
                ee_times_strehl=ee*strehl,
                negative_pixel_fraction=float(np.mean(psf < 0)))


def main(kind='psf', long=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=PSF_DATASETS if kind == 'psf' else ['WFSxy_DM_intensity_otf'])
    p.add_argument('--input-dir', type=Path, default=Path('output_infer'))
    p.add_argument('--test-dir', type=Path, default=Path('output_prep'))
    p.add_argument('--model-dir', type=Path, default=Path('output_ML'))
    p.add_argument('--output-dir', type=Path, default=Path('output_stats'))
    p.add_argument('--reference', type=Path, default=Path('dataInputs/ipsf_difflim.h5'))
    p.add_argument('--exposures', type=int, nargs='+', default=[10,100,1000,5000])
    p.add_argument('--ddof', type=int, choices=[0,1], default=1)
    p.add_argument('--dx', type=float, default=3.928571e-6)
    p.add_argument('--diameter', type=float, default=17.5e-3)
    p.add_argument('--wavelength', type=float, default=632.8e-9)
    p.add_argument('--pad', type=int, default=2)
    p.add_argument('--epsilon', type=float, default=1e-12)
    p.add_argument('--otf-truth-scale', type=float, default=1.0)
    p.add_argument('--otf-prediction-scale', type=float, default=1.0)
    p.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars')
    args = p.parse_args()
    args.long = long
    if min(args.dx,args.diameter,args.wavelength,args.epsilon) <= 0 or args.pad < 1:
        p.error('Optical parameters and epsilon must be positive; pad must be >= 1')
    exposures = sorted(set(args.exposures)) if long else [1]
    if not exposures or min(exposures) < 1:
        p.error('Exposures must be positive integers')
    with h5py.File(args.reference, 'r') as f:
        ref = image(first_dataset(f)[()])
    truth_path = args.test_dir / ('ao_ipsf_test_1.h5' if kind=='psf' else 'ao_iOTF_test_1.h5')
    for name in tqdm(args.datasets, desc='Datasets', unit='dataset', position=0,
                     dynamic_ncols=True, disable=args.no_progress):
        tqdm.write(f'Loading {name}...')
        folder = args.output_dir / (name + ('_long' if long else '_instant'))
        folder.mkdir(parents=True, exist_ok=True)
        records, summary, skipped = [], [], []
        with h5py.File(args.input_dir/name/'predictions.h5','r') as pf, h5py.File(truth_path,'r') as tf:
            pred, truth = first_dataset(pf), first_dataset(tf)
            if len(pred) != len(truth) or len(pred) == 0:
                raise ValueError(f'{name}: unequal or empty sample counts')
            shape = image(pred[0]).shape
            if kind == 'otf' and image(truth[0]).shape != shape:
                raise ValueError('OTF truth and prediction shapes must match; no OTF resampling is performed')
            c = context(ref, shape, kind, args)
            truth_scale, pred_scale = args.otf_truth_scale, args.otf_prediction_scale
            if kind == 'psf':
                vals = np.genfromtxt(args.model_dir/name/'max_val.txt', delimiter=',', dtype=None, encoding='utf-8')
                truth_scale, pred_scale = 1/float(vals[1][1]), 1.0
            for exposure in exposures:
                n_windows = len(pred)//exposure
                if n_windows == 0:
                    skipped.append(dict(exposure=exposure, reason='insufficient frames'))
                    tqdm.write(f'{name}: skipping exposure={exposure}; only {len(pred)} frames available')
                    continue
                group = []
                # Updating by frames keeps the bar responsive inside long-exposure windows.
                with tqdm(total=n_windows*exposure,
                          desc=f'{name} | {exposure} frame(s)/exposure',
                          unit='frame', position=1, leave=False,
                          dynamic_ncols=True, disable=args.no_progress) as progress:
                    for window in range(n_windows):
                        progress.set_postfix(window=f'{window+1}/{n_windows}', refresh=False)
                        start, stop = window*exposure, (window+1)*exposure
                        pa, ta = np.zeros(shape), np.zeros(shape)
                        for i in range(start, stop):
                            pa += image(pred[i])*pred_scale
                            ta += resize(image(truth[i]), shape)*truth_scale
                            if i < stop-1:
                                progress.update(1)
                        pa /= exposure
                        ta /= exposure
                        tm, pm = metrics(ta, kind, c, args), metrics(pa, kind, c, args)
                        row = dict(dataset=name, exposure_frames=exposure, window=window,
                                   start_frame=start, stop_frame_exclusive=stop)
                        for metric in tm:
                            t, v = tm[metric], pm[metric]
                            err = v-t
                            row.update({metric+'_truth':t, metric+'_prediction':v,
                                        metric+'_error':err,
                                        metric+'_relative_error_pct':100*divide(err,t,args.epsilon)})
                        records.append(row)
                        group.append(row)
                        progress.update(1)
                for metric in tm:
                    for series in ['truth','prediction','error','relative_error_pct']:
                        key = metric+'_'+series
                        values = np.array([r[key] for r in group])
                        s = describe(values, args.ddof)
                        finite = values[np.isfinite(values)]
                        entry = dict(dataset=name, exposure_frames=exposure, metric=metric,
                                     series=series, ddof=args.ddof, total_windows=n_windows,
                                     nonfinite_count=int(len(values)-len(finite)),
                                     discarded_frames=len(pred)%exposure, **s)
                        if series in ('error','relative_error_pct'):
                            entry.update(mae=float(np.mean(np.abs(finite))) if finite.size else np.nan,
                                         rmse=float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan)
                        summary.append(entry)
                tqdm.write(f'{name}: exposure={exposure}, windows={n_windows}, discarded={len(pred)%exposure}')
        tqdm.write(f'Writing CSV files for {name}...')
        write_csv(folder/'per_sample_metrics.csv', records)
        write_csv(folder/'summary_statistics.csv', summary)
        metadata = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        metadata.update(kind=kind, error_sign='prediction - truth',
                        windows='non-overlapping, incomplete final window discarded', skipped=skipped)
        (folder/'settings.json').write_text(json.dumps(metadata, indent=2))
        tqdm.write(f'Saved statistics: {folder}')
