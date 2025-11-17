import sys
from baseline import *
from bootstrap import *
from generateSyntheticData import *
from plots import *

np.random.seed(73)

# Experiment parameters
ALPHA = 0.05
NUMBERREPS = 20

# Bootstrap parameters
NUMBERBOOTSTRAP = 100
BLOCKLENGTH = 12 # rule of thumb: 1/(1 - phi)

# Data generation parameters
TIME = 200
PERIOD = 50
BASEPHI = 0.5
BASERHO = 0.5
STRENGTH = 0.05
NUMBERTRUE = 1
NUMBERCLUSTERS = 2
FIRMSPERCLUSTER = 3

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
    
    if command == "data":
        runGenerateData()
    elif command == "baseline":
        runBaseline()
    elif command == "bootstrap":
        runBootstrap()
    else:
        print(f"\nError: Unknown command '{command}'")
        printUsage()
        sys.exit(1)