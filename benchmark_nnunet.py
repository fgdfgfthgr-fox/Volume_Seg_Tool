import time
import os
from prepare_environment import run

if __name__ == "__main__":
    print('Start UroCell Dataset Validation using nnUNet')
    my_env = os.environ.copy()
    my_env["nnUNet_raw"] = "nnUNet_raw"
    my_env["nnUNet_preprocessed"] = "nnUNet_preprocessed"
    my_env["nnUNet_results"] = "nnUNet_results"
    start_time = time.time()
    run("nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity", custom_env=my_env)
    run("nnUNetv2_train 001 3d_fullres 0", custom_env=my_env)
    end_time_first_fold = time.time()
    print("Total time for a single fold of nnUNet training: ", end_time_first_fold - start_time)
    print("Estimated total time for 4 folds nnUNet training: ", 4 * (end_time_first_fold - start_time))