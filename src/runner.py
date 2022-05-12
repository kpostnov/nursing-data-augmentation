"""
File that gets executed!
Only import from experiments and tests
"""

import utils.settings as settings
import numpy as np


settings.init('sonar')

if __name__ == '__main__':
    import execute.sonar_lso_pipeline
