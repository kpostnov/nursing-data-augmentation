"""
File that gets executed!
Only import from experiments and tests
"""

import utils.settings as settings
import numpy as np


settings.init('pamap2')

if __name__ == '__main__':
    import execute.pamap2_lso_generation
