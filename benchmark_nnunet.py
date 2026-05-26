import time
from prepare_environment import run

if __name__ == "__main__":
    print('Start UroCell Dataset Validation using nnUNet')
    start_time = time.time()
    run("nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")
    run("nnUNetv2_train 001 3d_fullres 5")
    end_time = time.time()
    print("Total time for nnUNet training: ", end_time - start_time)