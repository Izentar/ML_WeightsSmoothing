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

def printStats(stat, metadata, **kwargs):
    """
        Deprecated
    """
    for st in stat:
        st.printPlots(startAt=startAt)

    avgStats = sf.averageStatistics(stat, relativeRootFolder=metadata.relativeRoot)
    avgStats.printPlots(**kwargs)

def printAvgStats(stat, metadataRoot, **kwargs):
    """
        Funkcja służy do wykonania uśredniania statystyk oraz rysowania wykresów z uśrednionych statystyk.
        metadataRoot - obiekt metadaty lub folder nadrzędny w którym zostaną zapisane logi. 
            Jeżeli folder nie istnieje, zostanie stworzony. W przypadku obiektu metadanych, logi zostaną zapisane zgodnie
            ze ścieżkami zawartymi w tym obiekcie.
        kwargs - argumenty dla rysowania wykresów
    """
    root = None
    if(isinstance(metadataRoot, sf.Metadata)):
        root = metadataRoot.relativeRoot
    elif(isinstance(metadataRoot, str)):
        root = metadataRoot
    else:
        raise Exception("Unknown data type: {}".format(type(metadataRoot)))
    avgStats = sf.averageStatistics(stat, relativeRootFolder=root)
    avgStats.printPlots(**kwargs)