import sys, os
from pathlib import Path

from experiments import setup
setup.run()

# zmień working directory na folder, w którym znajduje się ten plik.
#os.chdir(os.path.dirname(os.path.abspath(__file__)))

from framework.test.test_DefaultClasses import *
from framework.test.test_smoothingFramework import *
from framework.test.test_parser_exp import *

from framework import smoothingFramework as sf

import unittest, os

if __name__ == "__main__":
    sf.StaticData.LOG_FOLDER = os.path.join(os.getcwd(), 'framework', 'test', 'dump') 
    sf.StaticData.LOG_FOLDER = os.path.join(os.getcwd(), 'framework', 'test', 'dump')
    sf.StaticData.MAX_DEBUG_LOOPS = 50
    sf.CURRENT_STAT_PATH = sf.StaticData.LOG_FOLDER
    #sf.useDeterministic()
    unittest.main()
    #run()