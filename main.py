import sys
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
