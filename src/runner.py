"""
File that gets executed!
Only import from experiments and tests
"""
from utils import settings

# TESTS
# import tests.test_original_jens_windowize
# import tests.test_new_experiment_folder
# import tests.test_compare_model_input

# EXPERIMENTS
# import experiments.hello_world
# import experiments.opportunity_jens_cnn

# settings.init("sonar")
# import experiments.sonar_cnn
    


if __name__ == "__main__":
    import experiments.leave_subject_out
