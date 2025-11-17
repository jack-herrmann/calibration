import sys
from V2baseline import *
from V2bootstrap import *
from V2generateSyntheticData import *
from V2plots import (
    plotFWERvsDependence,
    plotPowerVsDependence,
    plotFWERvsPowerDetailed,
    plotFWERvsDependenceWithBootstrap,
    plotPowerVsDependenceWithBootstrap,
    plotFWERvsPowerWithBootstrap,
    plotDatasets
)

np.random.seed(73)

# Experiment parameters
ALPHA = 0.05
NUMBERREPS = 100

# Bootstrap parameters
NUMBERBOOTSTRAP = 1000
BLOCKLENGTH = 12 # rule of thumb: 1/(1 - phi)

# Data generation parameters
TIME = 1000
PERIOD = 50
BASEPHI = 0.5
BASERHO = 0.5
STRENGTH = 0.05
NUMBERTRUE = 2
NUMBERCLUSTERS = 2
FIRMSPERCLUSTER = 5

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

    table = createSummaryTableWithBootstrap(allResultsBootstrap)
    print(table.to_string(index=False))

    plotFWERvsDependenceWithBootstrap(allResultsBootstrap)
    plotPowerVsDependenceWithBootstrap(allResultsBootstrap)
    plotFWERvsPowerWithBootstrap(allResultsBootstrap)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        printUsage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "baseline":
        runBaseline()
    elif command == "bootstrap":
        runBootstrap()
    elif command == "data":
        runGenerateData()
    else:
        print(f"\nError: Unknown command '{command}'")
        printUsage()
        sys.exit(1)