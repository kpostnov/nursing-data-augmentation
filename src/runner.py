"""
File that gets executed!
Only import from experiments and tests
"""

import utils.settings as settings


settings.init('sonar')

if __name__ == '__main__':
    import execute.sonar_test
