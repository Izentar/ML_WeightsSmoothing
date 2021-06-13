import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from framework import smoothingFramework as sf
import traceback

def setupWorkingDir():
    os.chdir(Path(__file__).parents[1])

def printException(ex, types):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    sf.Output.printBash("Catched exception for '{}'. \nException type: {}\nFile name: {}:{}".format(
        types, exc_type, exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno
        ), 'err')
    sf.Output.printBash("Full traceback:\n{}".format(traceback.format_exc()), 'err')

def printStats(stat, metadata, startAt = -10):
    for st in stat:
        st.printPlots(startAt=startAt)

    avgStats = sf.averageStatistics(stat, relativeRootFolder=metadata.relativeRoot)
    avgStats.printPlots(startAt=startAt)