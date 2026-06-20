import numpy as np
import pandas as pd
import ast

def convert_value(s):
    s = s.strip()
    low = s.lower()

    if low == "true":
        return True
    if low == "false":
        return False

    try:
        return ast.literal_eval(s)
    except Exception:
        return s

def read_metadata_file(path):
    df = pd.read_csv(
        path,
        sep="=",
        comment="#",
        header=None,
        names=["key", "value"],
        engine="python",
        skip_blank_lines=True,
    )

    df["key"] = df["key"].str.strip()
    df["value"] = df["value"].str.split("#").str[0].str.strip()
    df = df[df["key"].notna() & df["value"].notna() & (df["key"] != "")]

    out = df.set_index("key")["value"].to_dict()
    return {k: convert_value(v) for k, v in out.items()}

def pixel_adder(data, scale_factor = [1,1], final_shape = None):
    """
    Generates a phase screen base on a Phase Spectrum Density (PSD)
    Parameters:
        data (ndarray): source data
        scale_factor (floar): scalling factor. Can be > or < o
        final_shape (ndarray): Option to input directly the desired shape. final_shape/data.shape must be an integer or the final shape will be data.shape*(final_shape//data.shape).
    Returns (list):
        D_phi (ndarray): 2D array representing the phase structure
    TODO: Error check
    """

    datashape = data.shape

    if final_shape == None:
        iscale = scale_factor[0]
        jscale = scale_factor[1]
        image_scaled = np.zeros((int(scale_factor[0]*datashape[0]), int(scale_factor[1]*datashape[1])))
    else:
        iscale = final_shape[0]/datashape[0]
        jscale = final_shape[1]/datashape[1]
        image_scaled = np.zeros((int(final_shape[0]), int(final_shape[1])))
    # print(image_scaled.shape)    
    for i in range(0, datashape[0]):
        for j in range(0, datashape[1]):
            iscaled = i*iscale
            jscaled = j*jscale
            image_scaled[int(iscaled):int(iscaled+iscale), int(jscaled):int(jscaled+jscale)] = data[i,j]
            # print(data[i,j])

    return image_scaled


def circle_to_square(positions_1d, Nact1D=11):
    dm_array = np.zeros((Nact1D, Nact1D))
    y, x = np.ogrid[:Nact1D, :Nact1D]
    center = (Nact1D - 1) / 2
    dist_sq = (x - center)**2 + (y - center)**2
    # Get indices of N closest points to center
    sorted_indices = np.argsort(dist_sq.flatten())
    active_indices = sorted_indices[:len(positions_1d)]
    dm_array.flat[active_indices] = positions_1d
    return dm_array

def eng_notation(value, decimals=1):
    """Engineering notation with custom decimal places"""
    if value == 0:
        return "0"
    
    expo = int(np.floor(np.log10(abs(value))))
    eng_expo = 3 * round(expo / 3)
    coeff = value / (10 ** eng_expo)
    
    # Use f-string with dynamic decimals
    return f"{coeff:.{decimals}f}e{eng_expo}"

def telescope_otf(nu_n, epsilon=None):
   """
   Computes a telescope's OTF
   Parameters:
       nu_n (ndarray): Normalized spatial/angular frequencies
       epsilon (float): obtruation ratio
   Returns (list):
       otf_tsc (ndarray): real 2D array representing the telescope OTF.
   TODO: Error check
   TODO: Implement anular pupils
   Based on "STEP-BY-STEP PROCEDURE FOR COMPUTING NUMERICALLY THE SEEING LIMITED POINT SPREAD FUNCTION FROM THE OPTICAL TURBULENCE PHASE STRUCTURE FUNCTION EQUATION" Laurent Jolissaint
    From PA 2025

   """
   # Plain pupil case (no central obtruation)
   otf_tsc = np.zeros_like(nu_n)
   mask = nu_n <= 1
   otf_tsc[mask] = (2/np.pi) * (np.arccos(nu_n[mask]) - nu_n[mask] * np.sqrt(1 - nu_n[mask]**2))
   return otf_tsc


def pad_center(arr, pad):
    """--------------------------------------------------
    Vectorised padding of arrays
    inputs:
        arr (ndarray): input array to be padded with the last 2 dimensions to be padded
        pad (uint): padding factor
    output:
        padded (ndarray): padded arrays
    --------------------------------------------------"""
    *batch, n, k = arr.shape
    if n != k:
        raise ValueError("last two dimensions must be square")
    
    M = n * pad
    out = np.zeros((*batch, M, M), dtype=arr.dtype)
    i0 = M // 2 - n // 2
    out[..., i0:i0+n, i0:i0+n] = arr  # https://www.geeksforgeeks.org/python/what-is-three-dots-or-ellipsis-in-python3/
    return out

def otf_from_psf(psf):
    """--------------------------------------------------
    Computes the Optical Transfer Function (OTF) from a Point Spread Function (PSF)
    inputs:
        psf (ndarray): input PSF(s) of shape (*batch, height, width)
    output:
        otf (ndarray): computed OTF
    --------------------------------------------------"""
    # Normalize across spatial dimensions (keep batch axis)
    psf = psf / psf.sum(axis=(-2, -1), keepdims=True)
    
    # Shift each PSF to origin, FFT over last 2 axes, shift back
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf, axes=(-2, -1)), axes=(-2, -1)),axes=(-2, -1))

def otf_to_psf(otf, psf_sum):
    """--------------------------------------------------
    Computes the Point Spread Function (PSF) from a Point Spread Function (OTF)
    inputs:
        otf (ndarray): input OTF(s) of shape (*batch, height, width)
    output:
        psf (ndarray): computed PSF(s) of shape (*batch, height, width)
    --------------------------------------------------"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(otf, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1)).real* psf_sum

def fwhm_1D(profile):
    """----------------------------------------------------
    Find FWHM for 1D profile: count of pixels >= half_max
    ----------------------------------------------------"""
    above_half = np.where(profile >= profile.max()/2)[0]
    if len(above_half) == 0:
        return 0
    return above_half[-1] - above_half[0] + 1  # +1 for inclusive count

def fwhm_2d(img):
    """----------------------------------------------------
    Measure FWHM directly on 2D PSF image without fitting.
    ----------------------------------------------------"""
    y_c, x_c = np.unravel_index(np.argmax(img, axis=None), img.shape) # https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html

    max_val = img[y_c, x_c]
    half_max = max_val / 2

    FWHM_x = fwhm_1D(img[y_c, :])
    FWHM_y = fwhm_1D(img[:, x_c])

    return FWHM_x, FWHM_y


def radial_average(psf, center=None):
    if center is None:
        center = np.unravel_index(np.argmax(psf), psf.shape)

    cy, cx = center
    y, x = np.indices(psf.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    r_max = int(r.max())
    radial_profile = []

    for i in range(r_max + 1):
        mask = (r >= i) & (r < i + 1)
        radial_profile.append(psf[mask].mean())

    return np.array(radial_profile)
