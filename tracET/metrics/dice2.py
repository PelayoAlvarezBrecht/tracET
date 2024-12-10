import scipy
import numpy as np
from tracET.core.skel import surface_skel,line_skel,point_skel
from tracET.core.diff import prepare_input





def cs_dice(tomo: np.ndarray, tomo_gt: np.ndarray,sigma=3,tomo_bin=False,tomo_imf=None,tomo_gt_bin=False,gt_imf=None, dilation=0) -> tuple:
    """
    Computes surface DICE metric (s-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogram (values >0 are considered foreground)
    :param tomo_gt: input ground truth (values >0 are considered foreground)
    :param sigma: Standard desviation for the gaussian filter (default 3)
    :param tomo_bin:True if the input predicted tomogram is a binary map (default False)
    :param tomo_imf: Threshold for filter masc to input predicted tomogram (default None)
    :param gt_bin: True if the input ground truth is a binary map (default False)
    :param gt_imf: Threshold for filter masc to input ground truth (default None)
    :param dilation: number of iterations to dilate the segmentation (default 0)
    :return: returns a 3-tuple where the 1st value is cl-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape

    # Getting segmentations ridges
    tomo_dsts=prepare_input(tomo,sigma,tomo_bin,tomo_imf).astype(np.float32)
    tomo_dsts = tomo_dsts * (tomo_dsts > 0)
    tomo_skel = surface_skel(tomo_dsts, f= 0)
    tomo_skel = tomo_skel * tomo #eliminte noise

    del tomo_dsts
    tomo_gt_dsts = prepare_input(tomo_gt, sigma, tomo_gt_bin, gt_imf).astype(np.float32)
    tomo_gt_dsts = tomo_gt_dsts * (tomo_gt_dsts > 0)
    tomo_gt_skel = surface_skel(tomo_gt_dsts, f= 0)
    tomo_gt_skel = tomo_gt_skel * tomo_gt #eliminte noise

    del tomo_gt_dsts

    # Dilation
    if dilation > 0:
        tomo_d = scipy.ndimage.binary_dilation(tomo, iterations=dilation)
        tomo_gt_d = scipy.ndimage.binary_dilation(tomo_gt, iterations=dilation)
    else:
        tomo_d = tomo
        tomo_gt_d = tomo_gt

    # Computing the metric
    tp = (tomo_skel * tomo_gt_d).sum() / tomo_skel.sum()
    ts = (tomo_gt_skel * tomo_d).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts, tomo_skel, tomo_gt_skel

def cl_dice(tomo: np.ndarray, tomo_gt: np.ndarray,sigma=3,tomo_bin=False,tomo_imf=None,tomo_gt_bin=False,gt_imf=None, dilation=0) -> tuple:
    """
    Computes centerline DICE metric (cl-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogram (values >0 are considered foreground)
    :param tomo_gt: input ground truth (values >0 are considered foreground)
    :param sigma: Standard desviation for the gaussian filter (default 3)
    :param tomo_bin:True if the input predicted tomogram is a binary map (default False)
    :param tomo_imf: Threshold for filter masc to input predicted tomogram (default None)
    :param gt_bin: True if the input ground truth is a binary map (default False)
    :param gt_imf: Threshold for filter masc to input ground truth (default None)
    :param dilation: number of iterations to dilate the segmentation (default 0)
    :return: returns a 3-tuple where the 1st value is cl-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape



    # Getting segmentations ridges
    tomo_dsts=prepare_input(tomo,sigma,tomo_bin,tomo_imf).astype(np.float32)
    tomo_dsts = tomo_dsts * (tomo_dsts > 0)
    tomo_gt_dsts = prepare_input(tomo_gt, sigma, tomo_gt_bin, gt_imf).astype(np.float32)
    tomo_gt_dsts = tomo_gt_dsts * (tomo_gt_dsts > 0)
    tomo_skel = line_skel(tomo_dsts, f=0.5)
    tomo_skel = tomo_skel * tomo #eliminte noise

    del tomo_dsts
    tomo_gt_skel= line_skel(tomo_gt_dsts, f=0.5)
    tomo_gt_skel = tomo_gt_skel * tomo_gt #eliminte noise

    del tomo_gt_dsts

    # Dilation
    if dilation > 0:
        tomo_d = scipy.ndimage.binary_dilation(tomo, iterations=dilation)
        tomo_gt_d = scipy.ndimage.binary_dilation(tomo_gt, iterations=dilation)
    else:
        tomo_d = tomo
        tomo_gt_d = tomo_gt

    # Computing the metric
    tp = (tomo_skel * tomo_gt_d).sum() / tomo_skel.sum()
    ts = (tomo_gt_skel * tomo_d).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts, tomo_skel, tomo_gt_skel

def pt_dice(tomo: np.ndarray, tomo_gt: np.ndarray,sigma=3,tomo_bin=False,tomo_imf=None,tomo_gt_bin=False,gt_imf=None, dilation=0) -> tuple:
    """
    Computes point DICE metric (pt-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogram (values >0 are considered foreground)
    :param tomo_gt: input ground truth (values >0 are considered foreground)
    :param sigma: Standard desviation for the gaussian filter (default 3)
    :param tomo_bin:True if the input predicted tomogram is a binary map (default False)
    :param tomo_imf: Threshold for filter masc to input predicted tomogram (default None)
    :param gt_bin: True if the input ground truth is a binary map (default False)
    :param gt_imf: Threshold for filter masc to input ground truth (default None)
    :param dilation: number of iterations to dilate the segmentation (default 0)
    :return: returns a 3-tuple where the 1st value is cl-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape


    # Getting segmentations ridges
    tomo_dsts=prepare_input(tomo,sigma,tomo_bin,tomo_imf).astype(np.float32)
    tomo_dsts=tomo_dsts*(tomo_dsts>0)
    tomo_gt_dsts = prepare_input(tomo_gt, sigma, tomo_gt_bin, gt_imf).astype(np.float32)
    tomo_gt_dsts = tomo_gt_dsts * (tomo_gt_dsts > 0)
    tomo_skel = point_skel(tomo_dsts, f=0.1)
    tomo_skel = tomo_skel * tomo #eliminte noise
    del tomo_dsts
    tomo_gt_skel = point_skel(tomo_gt_dsts, f=0.1)
    tomo_gt_skel = tomo_gt_skel * tomo_gt #eliminte noise
    del tomo_gt_dsts

    # Dilation
    if dilation > 0:
        tomo_d = scipy.ndimage.binary_dilation(tomo, iterations=dilation)
        tomo_gt_d = scipy.ndimage.binary_dilation(tomo_gt, iterations=dilation)
    else:
        tomo_d = tomo
        tomo_gt_d = tomo_gt

    # Computing the metric
    tp = (tomo_skel * tomo_gt_d).sum() / tomo_skel.sum()

    ts = (tomo_gt_skel * tomo_d).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts, tomo_skel, tomo_gt_skel
