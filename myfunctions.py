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