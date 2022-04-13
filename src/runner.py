"""
File that gets executed!
Only import from experiments and tests
"""
from utils import settings

# TESTS
# import tests.test_original_jens_windowize
# import tests.test_new_experiment_folder
<<<<<<< compare_model_input
=======
# import tests.test_compare_model_input
>>>>>>> main

# EXPERIMENTS
# import experiments.hello_world
# import experiments.opportunity_jens_cnn

<<<<<<< compare_model_input
if __name__ == "__main__":
    import experiments.leave_subject_out
    
=======
settings.init("sonar")

if __name__ == "__main__":
    import experiments.sonar_cnn
>>>>>>> main
