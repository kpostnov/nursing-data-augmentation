"""
File that gets executed!
Only import from experiments and tests
"""

import utils.settings as settings
import numpy as np


if __name__ == '__main__':
    settings.init('pamap2')
    import execute.pamap2_pipeline
