import sys, os
from pathlib import Path

# zmień working directory na folder, w którym znajduje się ten plik.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from framework.test.test_DefaultClasses import *
from framework.test.test_smoothingFramework import *
#from framework.test.test_experiments import *
import unittest


if __name__ == "__main__":
    #sf.useDeterministic()
    unittest.main()
    #run()