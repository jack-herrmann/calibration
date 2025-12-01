import sys
import os
from generateSyntheticData import *
from baseline import *
from bootstrap import *
from calibration_curves import *
from stability_analysis import *
from plots import *

def runGenerateData():
    clusteredDatasets = generateClusteredDatasets()
    plotDatasets(clusteredDatasets)

def runBaseline():
    allResults = runFullGrid()

    table = createSummaryTable(allResults)
    print(table.to_string(index=False))

    plotFWERvsDependence(allResults)
    plotPowerVsDependence(allResults)
    plotFWERvsPowerDetailed(allResults)

def runBootstrap():
    allResultsBootstrap = runFullGridWithBootstrap()

    table = createSummaryTableWithBootstrap(allResultsBootstrap)
    print(table.to_string(index=False))

    plotFWERvsDependenceWithBootstrap(allResultsBootstrap)
    plotPowerVsDependenceWithBootstrap(allResultsBootstrap)
    plotFWERvsPowerWithBootstrap(allResultsBootstrap)

def runCalibrationCurves():
    print("Scenario 1: High Time Dependence (phi=0.9, rho=0.5)")
    results_highphi = runCalibrationCurveExperiment(phi=0.9, rho=0.5)
    table1 = createCalibrationTable(results_highphi)
    print("\n" + table1.to_string(index=False))
    plotCalibrationCurves(results_highphi)

    print()

    print("Scenario 2: High Cross-Sectional Correlation (phi=0.5, rho=0.9)")
    results_highrho = runCalibrationCurveExperiment(phi=0.5, rho=0.9)
    table2 = createCalibrationTable(results_highrho)
    print("\n" + table2.to_string(index=False))
    plotCalibrationCurves(results_highrho)

    print()

    print("Scenario 3: No Dependence (phi=0.0, rho=0.0)")
    results_lowdep = runCalibrationCurveExperiment(phi=0.0, rho=0.0)
    table3 = createCalibrationTable(results_lowdep)
    print("\n" + table3.to_string(index=False))
    plotCalibrationCurves(results_lowdep)

def runStabilityAnalysis():
    print("Scenario 1: High Time Dependence (phi=0.9, rho=0.5)")
    results_highphi = runStabilityExperiment(phi=0.9, rho=0.5)
    table1 = createStabilityTable(results_highphi)
    print("\n" + table1.to_string(index=False))

    print()

    print("Scenario 2: High Cross-Sectional Correlation (phi=0.5, rho=0.9)")
    results_highrho = runStabilityExperiment(phi=0.5, rho=0.9)
    table2 = createStabilityTable(results_highrho)
    print("\n" + table2.to_string(index=False))

    print()

    print("Scenario 3: No Dependence (phi=0.0, rho=0.0)")
    results_lowdep = runStabilityExperiment(phi=0.0, rho=0.0)
    table3 = createStabilityTable(results_lowdep)
    print("\n" + table3.to_string(index=False))

def printUsage():
    """Print usage information for the main script."""
    print("\nUsage: python main.py <command>")
    print("\nAvailable commands:")
    print("  data         - Generate and plot synthetic datasets")
    print("  baseline     - Run baseline analysis with classical methods")
    print("  bootstrap    - Run bootstrap analysis")
    print("  calibration  - Run calibration curve experiments")
    print("  stability    - Run stability analysis")
    print("  all          - Run complete pipeline (data, calibration, stability) and save plots")
    print()

def runAll():
    """
    Run the complete pipeline: generate data, run stability and calibration analyses,
    and save all plots to the plots folder.
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    print("=" * 60)
    print("RUNNING COMPLETE ANALYSIS PIPELINE")
    print("=" * 60)
    print()

    # 1. Generate and save synthetic data plots
    print("Step 1: Generating synthetic data...")
    clusteredDatasets = generateClusteredDatasets()
    plotDatasets(clusteredDatasets, savePath="plots/synthetic_datasets.png")
    print()

    # 2. Run calibration curves for three scenarios
    print("Step 2: Running calibration curve experiments...")
    print()

    print("Scenario 1: High Time Dependence (phi=0.9, rho=0.5)")
    results_highphi_cal = runCalibrationCurveExperiment(phi=0.9, rho=0.5)
    table1 = createCalibrationTable(results_highphi_cal)
    print("\n" + table1.to_string(index=False))
    plotCalibrationCurves(results_highphi_cal, savePath="plots/calibration_highphi.png")
    print()

    print("Scenario 2: High Cross-Sectional Correlation (phi=0.5, rho=0.9)")
    results_highrho_cal = runCalibrationCurveExperiment(phi=0.5, rho=0.9)
    table2 = createCalibrationTable(results_highrho_cal)
    print("\n" + table2.to_string(index=False))
    plotCalibrationCurves(results_highrho_cal, savePath="plots/calibration_highrho.png")
    print()

    print("Scenario 3: No Dependence (phi=0.0, rho=0.0)")
    results_lowdep_cal = runCalibrationCurveExperiment(phi=0.0, rho=0.0)
    table3 = createCalibrationTable(results_lowdep_cal)
    print("\n" + table3.to_string(index=False))
    plotCalibrationCurves(results_lowdep_cal, savePath="plots/calibration_lowdep.png")
    print()

    # 3. Run stability analysis for three scenarios
    print("Step 3: Running stability analysis...")
    print()

    print("Scenario 1: High Time Dependence (phi=0.9, rho=0.5)")
    results_highphi_stab = runStabilityExperiment(phi=0.9, rho=0.5)
    table4 = createStabilityTable(results_highphi_stab)
    print("\n" + table4.to_string(index=False))
    print()

    print("Scenario 2: High Cross-Sectional Correlation (phi=0.5, rho=0.9)")
    results_highrho_stab = runStabilityExperiment(phi=0.5, rho=0.9)
    table5 = createStabilityTable(results_highrho_stab)
    print("\n" + table5.to_string(index=False))
    print()

    print("Scenario 3: No Dependence (phi=0.0, rho=0.0)")
    results_lowdep_stab = runStabilityExperiment(phi=0.0, rho=0.0)
    table6 = createStabilityTable(results_lowdep_stab)
    print("\n" + table6.to_string(index=False))
    print()

    print("=" * 60)
    print("PIPELINE COMPLETE - All plots saved to 'plots/' directory")
    print("=" * 60)



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
    elif command == "all":
        runAll()
    else:
        print(f"\nError: Unknown command '{command}'")
        printUsage()
        sys.exit(1)
