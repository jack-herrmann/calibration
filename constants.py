"""
Configuration constants for the multiple testing calibration experiment.

This module contains all experiment parameters organized by category.
"""

# Experiment parameters
ALPHA = 0.05
ALPHALEVELS = [0.005, 0.01, 0.02, 0.05]  # For calibration curves and stability
NUMBERREPS = 100
NUMBERREPS_STABILITY = 50  # fewer reps for stability (computationally expensive)

# Bootstrap parameters
NUMBERBOOTSTRAP = 100
BLOCKLENGTH = 12  # rule of thumb: 1/(1 - phi)

# Data generation parameters
TIME = 200
PERIOD = 50
BASEPHI = 0.5
BASERHO = 0.5
STRENGTH = 0.15
NUMBERTRUE = 1
NUMBERCLUSTERS = 2
FIRMSPERCLUSTER = 3
