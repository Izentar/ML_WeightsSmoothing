from framework import smoothingFramework as sf

def printException(ex, types):
    sf.Output.printBash("Catched exception for '{}'. Exception type: {}".format(types, ex.message if hasattr(ex, 'message') else "None"), 'err')
