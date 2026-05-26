import time
from prepare_environment import run

if __name__ == "__main__":
    print('Start UroCell Dataset Training using VST')
    start_time = time.time()
    run("python pl_module_dit.py")
    end_time = time.time()
    print("Total time for VST training: ", end_time - start_time)