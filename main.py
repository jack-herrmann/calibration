import sys
from generateSyntheticData import *
from baseline import *
from bootstrap import *
from calibration_curves import *
from stability_analysis import *
from plots import *

np.random.seed(73)

# Experiment parameters
ALPHA = 0.05
ALPHALEVELS = [0.005, 0.01, 0.02, 0.05]  # For calibration curves and stability
NUMBERREPS = 100
NUMBERREPS_STABILITY = 50  # fewer reps for stability (computationally expensive)

# Bootstrap parameters
NUMBERBOOTSTRAP = 100
BLOCKLENGTH = 12 # rule of thumb: 1/(1 - phi)

# Data generation parameters
TIME = 200
PERIOD = 50
BASEPHI = 0.5
BASERHO = 0.5
STRENGTH = 0.15
NUMBERTRUE = 1
NUMBERCLUSTERS = 2
FIRMSPERCLUSTER = 3


def printUsage():
    print("Usage: python main.py [data|baseline|bootstrap|calibration|stability]")
    print("  data        - Generate and visualize synthetic datasets")
    print("  baseline    - Run baseline methods (Bonferroni, Holm, BH)")
    print("  bootstrap   - Run bootstrap-based methods")
    print("  calibration - Run calibration curve experiments across multiple alpha levels")
    print("  stability   - Run discovery stability analysis across bootstrap resamples")

def runGenerateData():
    clusteredDatasets = generateClusteredDatasets(
        TIME,
        NUMBERCLUSTERS,
        FIRMSPERCLUSTER,
        NUMBERTRUE,
        STRENGTH,
        BASEPHI,
        BASERHO
    )
    
    plotDatasets(clusteredDatasets, PERIOD)

def runBaseline():
    allResults = runFullGrid(
        NUMBERREPS,
        ALPHA,
        TIME,
        NUMBERCLUSTERS,
        FIRMSPERCLUSTER,
        NUMBERTRUE,
        STRENGTH,
        BASEPHI,
        BASERHO
    )

    table = createSummaryTable(allResults, ALPHA)
    print(table.to_string(index=False))

    plotFWERvsDependence(allResults)
    plotPowerVsDependence(allResults)
    plotFWERvsPowerDetailed(allResults)

def runBootstrap():
    allResultsBootstrap = runFullGridWithBootstrap(
        NUMBERREPS,
        ALPHA,
        NUMBERBOOTSTRAP,
        BLOCKLENGTH,
        TIME,
        NUMBERCLUSTERS,
        FIRMSPERCLUSTER,
        NUMBERTRUE,
        STRENGTH,
        BASEPHI,
        BASERHO
    )

    table = createSummaryTableWithBootstrap(allResultsBootstrap, ALPHA)
    print(table.to_string(index=False))

    plotFWERvsDependenceWithBootstrap(allResultsBootstrap)
    plotPowerVsDependenceWithBootstrap(allResultsBootstrap)
    plotFWERvsPowerWithBootstrap(allResultsBootstrap)

def runCalibrationCurves():
    print("Scenario 1: High Time Dependence (φ=0.9, ρ=0.5)")
    results_highphi = runCalibrationCurveExperiment(
        alphaLevels=ALPHALEVELS,
        numReps=NUMBERREPS,
        numberBootstrap=NUMBERBOOTSTRAP,
        blockLength=BLOCKLENGTH,
        time=TIME,
        numberClusters=NUMBERCLUSTERS,
        firmsPerCluster=FIRMSPERCLUSTER,
        numberTrue=NUMBERTRUE,
        strength=STRENGTH,
        phi=0.9,
        rho=0.5
    )
    table1 = createCalibrationTable(results_highphi)
    print("\n" + table1.to_string(index=False))
    plotCalibrationCurves(results_highphi)

    print()

    print("Scenario 2: High Cross-Sectional Correlation (φ=0.5, ρ=0.9)")
    results_highrho = runCalibrationCurveExperiment(
        alphaLevels=ALPHALEVELS,
        numReps=NUMBERREPS,
        numberBootstrap=NUMBERBOOTSTRAP,
        blockLength=BLOCKLENGTH,
        time=TIME,
        numberClusters=NUMBERCLUSTERS,
        firmsPerCluster=FIRMSPERCLUSTER,
        numberTrue=NUMBERTRUE,
        strength=STRENGTH,
        phi=0.5,
        rho=0.9
    )
    table2 = createCalibrationTable(results_highrho)
    print("\n" + table2.to_string(index=False))
    plotCalibrationCurves(results_highrho)

    print()

    print("Scenario 3: Low Dependence (φ=0.0, ρ=0.0)")
    results_lowdep = runCalibrationCurveExperiment(
        alphaLevels=ALPHALEVELS,
        numReps=NUMBERREPS,
        numberBootstrap=NUMBERBOOTSTRAP,
        blockLength=BLOCKLENGTH,
        time=TIME,
        numberClusters=NUMBERCLUSTERS,
        firmsPerCluster=FIRMSPERCLUSTER,
        numberTrue=NUMBERTRUE,
        strength=STRENGTH,
        phi=0.0,
        rho=0.0
    )
    table3 = createCalibrationTable(results_lowdep)
    print("\n" + table3.to_string(index=False))
    plotCalibrationCurves(results_lowdep)

def runStabilityAnalysis():
    print("Scenario 1: High Time Dependence (φ=0.9, ρ=0.5)")
    results_highphi = runStabilityExperiment(
        alphaLevels=ALPHALEVELS,
        numReps=NUMBERREPS_STABILITY,
        numberBootstrap=NUMBERBOOTSTRAP,
        blockLength=BLOCKLENGTH,
        time=TIME,
        numberClusters=NUMBERCLUSTERS,
        firmsPerCluster=FIRMSPERCLUSTER,
        numberTrue=NUMBERTRUE,
        strength=STRENGTH,
        phi=0.9,
        rho=0.5
    )
    table1 = createStabilityTable(results_highphi)
    print("\n" + table1.to_string(index=False))

    print()

    print("Scenario 2: High Cross-Sectional Correlation (φ=0.5, ρ=0.9)")
    results_highrho = runStabilityExperiment(
        alphaLevels=ALPHALEVELS,
        numReps=NUMBERREPS_STABILITY,
        numberBootstrap=NUMBERBOOTSTRAP,
        blockLength=BLOCKLENGTH,
        time=TIME,
        numberClusters=NUMBERCLUSTERS,
        firmsPerCluster=FIRMSPERCLUSTER,
        numberTrue=NUMBERTRUE,
        strength=STRENGTH,
        phi=0.5,
        rho=0.9
    )
    table2 = createStabilityTable(results_highrho)
    print("\n" + table2.to_string(index=False))

    print()

    print("Scenario 3: Low Dependence (φ=0.0, ρ=0.0)")
    results_lowdep = runStabilityExperiment(
        alphaLevels=ALPHALEVELS,
        numReps=NUMBERREPS_STABILITY,
        numberBootstrap=NUMBERBOOTSTRAP,
        blockLength=BLOCKLENGTH,
        time=TIME,
        numberClusters=NUMBERCLUSTERS,
        firmsPerCluster=FIRMSPERCLUSTER,
        numberTrue=NUMBERTRUE,
        strength=STRENGTH,
        phi=0.0,
        rho=0.0
    )
    table3 = createStabilityTable(results_lowdep)
    print("\n" + table3.to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        printUsage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "data":
        runGenerateData()
    elif command == "baseline":
        runBaseline()
    elif command == "bootstrap":
        runBootstrap()
    elif command == "calibration":
        runCalibrationCurves()
    elif command == "stability":
        runStabilityAnalysis()
    else:
        print(f"\nError: Unknown command '{command}'")
        printUsage()
        sys.exit(1)
