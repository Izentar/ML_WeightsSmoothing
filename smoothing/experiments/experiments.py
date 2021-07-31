import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from framework import smoothingFramework as sf
import traceback

for i, arg in enumerate(sys.argv):
    if(arg == "--debug" or arg == "--test"):
        sf.StaticData.TEST_MODE = True

##############################################################

def setupWorkingDir():
    os.chdir(Path(__file__).parents[1])

def printException(ex, types):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    sf.Output.printBash("Catched exception for '{}'. \nException type: {}\nFile name: {}:{}".format(
        types, exc_type, exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno
        ), 'err')
    sf.Output.printBash("Full traceback:\n{}".format(traceback.format_exc()), 'err')

def printStats(stat, metadata, startAt = -10, fileFormat='.svg', dpi=900):
    for st in stat:
        st.printPlots(startAt=startAt)

    avgStats = sf.averageStatistics(stat, relativeRootFolder=metadata.relativeRoot)
    avgStats.printPlots(startAt=startAt, fileFormat=fileFormat, dpi=dpi)

def printAvgStats(stat, metadataRoot, startAt = -10, runningAvgSize=1, fileFormat='.svg', dpi=900):
    """
        metadataRoot - obiekt metadaty lub folder nadrzędny w którym zostaną zapisane logi. Jeżeli folder nie istnieje, zostanie stworzony.
    """
    root = None
    if(isinstance(metadataRoot, sf.Metadata)):
        root = metadataRoot.relativeRoot
    elif(isinstance(metadataRoot, str)):
        root = metadataRoot
    else:
        raise Exception("Unknown data type: {}".format(type(metadataRoot)))
    avgStats = sf.averageStatistics(stat, relativeRootFolder=root)
    avgStats.printPlots(startAt=startAt, runningAvgSize=runningAvgSize, fileFormat=fileFormat, dpi=dpi)