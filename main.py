import sys
import os
from generateSyntheticData import *
from baseline import *
from bootstrap import *
from calibration_curves import *
from stability_analysis import *
from plots import *
from constants import SCENARIOS

def runGenerateData():
    clusteredDatasets = generateClusteredDatasets()
    plotDatasets(clusteredDatasets)

def runBaseline():
    allResults = runFullGrid()
    createSummaryTable(allResults, savePath="results/baseline_grid_summary.csv")

    plotFWERvsDependence(allResults)
    plotPowerVsDependence(allResults)
    plotFWERvsPowerDetailed(allResults)

def runBootstrap():
    allResultsBootstrap = runFullGridWithBootstrap()
    createSummaryTableWithBootstrap(allResultsBootstrap, savePath="results/bootstrap_grid_summary.csv")

    plotFWERvsDependenceWithBootstrap(allResultsBootstrap)
    plotPowerVsDependenceWithBootstrap(allResultsBootstrap)
    plotFWERvsPowerWithBootstrap(allResultsBootstrap)

def runCalibrationCurves():
    scenario_names = ['worstcase', 'highphi', 'highrho', 'baseline']

    for idx, (phi, rho, description) in enumerate(SCENARIOS, 1):
        print(f"Scenario {idx}: {description} (phi={phi}, rho={rho})")
        results = runCalibrationCurveExperiment(phi=phi, rho=rho)
        createCalibrationTable(results, savePath=f"results/calibration_{scenario_names[idx-1]}.csv")
        plotCalibrationCurves(results)
        print()

def runStabilityAnalysis():
    scenario_names = ['worstcase', 'highphi', 'highrho', 'baseline']

    for idx, (phi, rho, description) in enumerate(SCENARIOS, 1):
        print(f"Scenario {idx}: {description} (phi={phi}, rho={rho})")
        results = runStabilityExperiment(phi=phi, rho=rho)
        createStabilityTable(results, savePath=f"results/stability_{scenario_names[idx-1]}.csv")
        print()

def printUsage():
    """Print usage information for the main script"""
    print("\nUsage: python main.py <command>")
    print("\nAvailable commands:")
    print("  data         - Generate and plot synthetic datasets")
    print("  baseline     - Run baseline analysis with classical methods")
    print("  bootstrap    - Run bootstrap analysis")
    print("  calibration  - Run calibration curve experiments")
    print("  stability    - Run stability analysis")
    print("  all          - Run complete pipeline (data, bootstrap, calibration, stability) and save all plots")
    print()

def runAll():
    """Run the complete pipeline and save all plots and tables"""
    # Create output directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Step 1: Generating synthetic data...")
    clusteredDatasets = generateClusteredDatasets()
    plotDatasets(clusteredDatasets, savePath="plots/synthetic_datasets.png")
    print()

    # 2. Run bootstrap grid analysis
    print("Step 2: Running bootstrap grid analysis...")
    print("(Testing all methods across parameter space)")
    allResultsBootstrap = runFullGridWithBootstrap()
    createSummaryTableWithBootstrap(allResultsBootstrap, savePath="results/bootstrap_grid_summary.csv")
    print()

    plotFWERvsDependenceWithBootstrap(allResultsBootstrap, savePath="plots/bootstrap_fwer_vs_dependence.png")
    plotPowerVsDependenceWithBootstrap(allResultsBootstrap, savePath="plots/bootstrap_power_vs_dependence.png")
    plotFWERvsPowerWithBootstrap(allResultsBootstrap, savePath="plots/bootstrap_fwer_vs_power.png")
    print()

    # 3. Run calibration curves for key scenarios
    print("Step 3: Running calibration curve experiments...")
    print("(Testing calibration at key scenarios)")
    print()

    scenario_names = ['worstcase', 'highphi', 'highrho', 'baseline']
    for idx, (phi, rho, description) in enumerate(SCENARIOS, 1):
        print(f"Scenario {idx}: {description} (phi={phi}, rho={rho})")
        results = runCalibrationCurveExperiment(phi=phi, rho=rho)
        createCalibrationTable(results, savePath=f"results/calibration_{scenario_names[idx-1]}.csv")
        plotCalibrationCurves(results, savePath=f"plots/calibration_{scenario_names[idx-1]}.png")
        print()

    # 4. Run stability analysis for key scenarios
    print("Step 4: Running stability analysis...")
    print("(Testing discovery stability at key scenarios)")
    print()

    for idx, (phi, rho, description) in enumerate(SCENARIOS, 1):
        print(f"Scenario {idx}: {description} (phi={phi}, rho={rho})")
        results = runStabilityExperiment(phi=phi, rho=rho)
        createStabilityTable(results, savePath=f"results/stability_{scenario_names[idx-1]}.csv")
        print()

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("  Plots saved to: plots/")
    print("  Tables saved to: results/")
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
