"""
File that gets executed!
Only import from experiments and tests
"""

import utils.settings as settings


if __name__ == '__main__':
	settings.init('opportunity')
	import execute.pamap2_test
