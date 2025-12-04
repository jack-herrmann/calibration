import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from generateSyntheticData import *
from baseline import *
from bootstrap import *
from calibration_curves import *
from stability_analysis import *
from plots import *
from constants import SCENARIOS

# Helper functions for parallel execution
def _run_calibration_scenario(args):
    """Helper function to run a single calibration scenario (for parallel execution)"""
    idx, phi, rho, description, scenario_name = args
    print(f"[Calibration {idx}] Starting: {description} (phi={phi}, rho={rho})")
    results = runCalibrationCurveExperiment(phi=phi, rho=rho)
    createCalibrationTable(results, savePath=f"results/calibration_{scenario_name}.csv")
    plotCalibrationCurves(results, savePath=f"plots/calibration_{scenario_name}.png")
    print(f"[Calibration {idx}] ✓ Complete: {scenario_name}")
    return scenario_name

def _run_stability_scenario(args):
    """Helper function to run a single stability scenario (for parallel execution)"""
    idx, phi, rho, description, scenario_name = args
    print(f"[Stability {idx}] Starting: {description} (phi={phi}, rho={rho})")
    results = runStabilityExperiment(phi=phi, rho=rho)
    createStabilityTable(results, savePath=f"results/stability_{scenario_name}.csv")
    print(f"[Stability {idx}] ✓ Complete: {scenario_name}")
    return scenario_name

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

    print("Step 1: Generating synthetic data...")
    clusteredDatasets = generateClusteredDatasets()
    plotDatasets(clusteredDatasets, savePath="plots/synthetic_datasets.png")
    print()

    # 2. Run bootstrap grid analysis (PARALLEL)
    print("Step 2: Running bootstrap grid analysis in parallel...")
    print("(Testing 8 scenarios: 4 phi sweeps + 4 rho sweeps concurrently)")
    allResultsBootstrap = runFullGridWithBootstrapParallel()
    createSummaryTableWithBootstrap(allResultsBootstrap, savePath="results/bootstrap_grid_summary.csv")
    print(f"✓ Bootstrap grid analysis complete")
    print()

    plotFWERvsDependenceWithBootstrap(allResultsBootstrap, savePath="plots/bootstrap_fwer_vs_dependence.png")
    plotPowerVsDependenceWithBootstrap(allResultsBootstrap, savePath="plots/bootstrap_power_vs_dependence.png")
    plotFWERvsPowerWithBootstrap(allResultsBootstrap, savePath="plots/bootstrap_fwer_vs_power.png")
    print()

    # 3. Run calibration curves for key scenarios (PARALLEL)
    print("Step 3: Running calibration curve experiments in parallel...")
    print(f"(Running {len(SCENARIOS)} scenarios concurrently)")
    print()

    scenario_names = ['worstcase', 'highphi', 'highrho', 'baseline']
    calibration_tasks = [
        (idx, phi, rho, description, scenario_names[idx-1])
        for idx, (phi, rho, description) in enumerate(SCENARIOS, 1)
    ]

    with ProcessPoolExecutor(max_workers=len(SCENARIOS)) as executor:
        futures = [executor.submit(_run_calibration_scenario, task) for task in calibration_tasks]
        for future in as_completed(futures):
            scenario_name = future.result()
            # Results already logged by worker

    print(f"✓ All {len(SCENARIOS)} calibration scenarios complete")
    print()

    # 4. Run stability analysis for key scenarios (PARALLEL)
    print("Step 4: Running stability analysis in parallel...")
    print(f"(Running {len(SCENARIOS)} scenarios concurrently)")
    print()

    stability_tasks = [
        (idx, phi, rho, description, scenario_names[idx-1])
        for idx, (phi, rho, description) in enumerate(SCENARIOS, 1)
    ]

    with ProcessPoolExecutor(max_workers=len(SCENARIOS)) as executor:
        futures = [executor.submit(_run_stability_scenario, task) for task in stability_tasks]
        for future in as_completed(futures):
            scenario_name = future.result()
            # Results already logged by worker

    print(f"✓ All {len(SCENARIOS)} stability scenarios complete")
    print()

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("  Plots saved to: plots/")
    print("  Tables saved to: results/")
    print("=" * 60)

def runStep4():

    # 4. Run stability analysis for key scenarios (PARALLEL)
    print("Step 4: Running stability analysis in parallel...")
    print(f"(Running {len(SCENARIOS)} scenarios concurrently)")
    print()

    stability_tasks = [
        (idx, phi, rho, description, scenario_names[idx-1])
        for idx, (phi, rho, description) in enumerate(SCENARIOS, 1)
    ]

    with ProcessPoolExecutor(max_workers=len(SCENARIOS)) as executor:
        futures = [executor.submit(_run_stability_scenario, task) for task in stability_tasks]
        for future in as_completed(futures):
            scenario_name = future.result()
            # Results already logged by worker

    print(f"✓ All {len(SCENARIOS)} stability scenarios complete")
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
    elif command == "four":
        runStep4()
    else:
        print(f"\nError: Unknown command '{command}'")
        printUsage()
        sys.exit(1)
