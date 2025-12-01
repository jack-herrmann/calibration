"""
Configuration constants for the multiple testing calibration experiment.

This module contains all experiment parameters organized by category.
"""

import numpy as np

# Experiment parameters
ALPHA = 0.05
ALPHALEVELS = [0.005, 0.01, 0.02, 0.05]  # For calibration curves and stability
NUMBERREPS = 100
NUMBERREPS_STABILITY = 50  # fewer reps for stability (computationally expensive)

# Bootstrap parameters
NUMBERBOOTSTRAP = 100
MIN_BLOCKLENGTH = 2  # minimum block length to avoid degenerate cases
MAX_BLOCKLENGTH = 50  # maximum to avoid extremely long blocks

# Data generation parameters
TIME = 200
PERIOD = 50
BASEPHI = 0.5
BASERHO = 0.5
STRENGTH = 0.15
NUMBERTRUE = 1
NUMBERCLUSTERS = 2
FIRMSPERCLUSTER = 3

def computeBlockLength(phi):
    """
    Compute optimal block length for moving block bootstrap based on AR(1) coefficient.

    Uses the rule of thumb: block_length ≈ 1/(1-phi)

    Args:
        phi: AR(1) coefficient (time dependence parameter)

    Returns:
        int: Optimal block length, bounded by MIN_BLOCKLENGTH and MAX_BLOCKLENGTH
    """
    if phi >= 1.0:
        return MAX_BLOCKLENGTH
    if phi <= 0.0:
        return MIN_BLOCKLENGTH

    # Rule of thumb: 1/(1-phi)
    optimal = 1.0 / (1.0 - phi)

    # Round to nearest integer and bound
    block_length = int(np.round(optimal))
    block_length = max(MIN_BLOCKLENGTH, min(MAX_BLOCKLENGTH, block_length))

    return block_length
