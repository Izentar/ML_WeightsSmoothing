import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parents[0]))

from framework.test.test_DefaultClasses import *
from framework.test.test_smoothingFramework import *
#from framework.test.test_experiments import *
import unittest


if __name__ == "__main__":
    #sf.useDeterministic()
    unittest.main()
    #run()