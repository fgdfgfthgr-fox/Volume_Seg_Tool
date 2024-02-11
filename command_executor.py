import subprocess
import psutil

# Taken from kohya_ss [https://github.com/bmaltais/kohya_ss]

class CommandExecutor:
    def __init__(self):
        self.process = None

    def execute_command(self, run_cmd):
        if self.process and self.process.poll() is None:
            print('The command is already running. Please wait for it to finish.')
        else:
            self.process = subprocess.Popen(run_cmd, shell=True)

    def kill_command(self):
        if self.process and self.process.poll() is None:
            try:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                print('The running process has been terminated.')
            except psutil.NoSuchProcess:
                print('The process does not exist.')
            except Exception as e:
                print(f'Error when terminating process: {e}')
        else:
            print('There is no running process to kill.')