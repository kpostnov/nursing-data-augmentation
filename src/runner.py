"""
File that gets executed!
"""

import argparse
import utils.settings as settings
import execute.pamap2_lso_generation as pamap2_lso_generation
import execute.pamap2_lso_evaluation as pamap2_lso_evaluation
import execute.sonar_lso_generation as sonar_lso_generation
import execute.sonar_lso_evaluation as sonar_lso_evaluation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", 
                        choices=['pamap2', 'sonar', 'sonar_lab'], 
                        type=str, 
                        default="pamap2", 
                        help="Dataset to use")                 
    parser.add_argument("--mode", choices=['gen', 'eval'], type=str, default="gen", help="Mode to use")
    parser.add_argument("--data_path", type=str, default="", help="Path to the dataset directory")
    parser.add_argument("--synth_data_path", type=str, default="", help="Path to directory where the generated data is stored")
    parser.add_argument("--random_data_path", 
                        type=str, 
                        default="", 
                        help="Path to random data file (used for evaluation)")
    parser.add_argument("--window_size", type=int, help="Window size")
    parser.add_argument("--stride_size", type=int, help="Stride size")

    args = parser.parse_args()

    # Set default window and stride size if not specified (depending on dataset)
    if args.dataset == "pamap2":
        if args.window_size is None:
            args.window_size = 100
        if args.stride_size is None:
            args.stride_size = args.window_size
    else:
        if args.window_size is None:
            args.window_size = 300
        if args.stride_size is None:
            args.stride_size = args.window_size

    # Initialize dataset-specific settings
    settings.init(args)

    # Execute the specified action
    if args.dataset == "pamap2":
        if args.mode == "gen":
            pamap2_lso_generation.start_generation()
        elif args.mode == "eval":
            pamap2_lso_evaluation.start_evaluation()
        else:
            raise Exception("Unknown mode")    
    elif args.dataset == "sonar" or args.dataset == "sonar_lab":
        if args.mode == "gen":
            sonar_lso_generation.start_generation()
        elif args.mode == "eval":
            sonar_lso_evaluation.start_evaluation()
        else:
            raise Exception("Unknown mode")    
    else:
        raise Exception("Unknown dataset")
