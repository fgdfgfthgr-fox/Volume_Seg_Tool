
import re
import subprocess
import os
import sys

if sys.version_info < (3, 9):
    print("ERROR: Python 3.9 or higher is required.")
    sys.exit(1)

script_path = os.path.dirname(os.path.realpath(__file__))

python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")

# Most of the code were copied from Stable Diffusion WebUI setup code...


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = True) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def run_pip(command, desc=None, live=True):

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.7.1 torchvision --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements.txt")

    #print(f"Python {sys.version}")

    run(f"{torch_command}", "Installing torch and torchvision", "Couldn't install torch", live=True)
    run(f"pip install packaging --trusted-host pypi.org --trusted-host files.pythonhosted.org", "Installing packaging", "Couldn't install packaging", live=True)
    requirements_file = os.path.join(script_path, requirements_file)
    #if not requirements_met(requirements_file):
    run_pip(f"install -r \"{requirements_file}\" -U", "requirements")


if __name__ == "__main__":
    prepare_environment()
    print("Installation Completed")
