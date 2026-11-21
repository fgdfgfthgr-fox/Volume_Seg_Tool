import re
import subprocess
import os
import sys

if sys.version_info < (3, 11):
    print("ERROR: Python 3.11 or higher is required.")
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


def detect_linux_pywebview_backend():
    """Detect the best pywebview backend for this Linux environment.

    Priority: explicit PYWEBVIEW_BACKEND override > GTK (if importable) > Qt.

    GTK requires PyGObject (gi) and WebKit2 to be importable *from the current
    interpreter*. A normal venv cannot see system PyGObject, so GTK only
    works here if the venv was created with --system-site-packages (or if
    PyGObject was pip-installed into the venv). Qt, by contrast, is pure pip
    and always works, so it is the fallback.
    """
    override = os.environ.get('PYWEBVIEW_BACKEND', '').strip()

    gtk_available = False
    try:
        import gi
        try:
            gi.require_version('Gtk', '3.0')
            gi.require_version('WebKit2', '4.1')
            gtk_available = True
        except ValueError:
            try:
                gi.require_version('WebKit2', '4.0')
                gtk_available = True
            except ValueError:
                pass
    except ImportError:
        pass

    if override:
        if override == 'gtk' and not gtk_available:
            print("WARNING: PYWEBVIEW_BACKEND=gtk was requested but gi/WebKit2 "
                  "is not importable from this python. Falling back to qt.")
            return 'qt'
        return override

    return 'gtk' if gtk_available else 'qt'


def install_pywebview():
    """Install the pywebview backend extras appropriate for this platform.

    - Windows (EdgeChromium) and macOS (WKWebView) are covered by the base
      pywebview install from requirements.txt, so nothing extra is needed.
    - Linux requires either the GTK or the Qt backend to be added explicitly.
    """
    if not sys.platform.startswith('linux'):
        return

    backend = detect_linux_pywebview_backend()
    print(f"Installing pywebview with '{backend}' backend on Linux")
    run_pip(f"install pywebview[{backend}]", f"pywebview[{backend}]")


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.14.1 torchvision --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements.txt")

    #print(f"Python {sys.version}")

    run(f"{torch_command}", "Installing torch and torchvision", "Couldn't install torch", live=True)
    run(f"pip install packaging --trusted-host pypi.org --trusted-host files.pythonhosted.org", "Installing packaging", "Couldn't install packaging", live=True)
    requirements_file = os.path.join(script_path, requirements_file)
    #if not requirements_met(requirements_file):
    run_pip(f"install -r \"{requirements_file}\" -U", "requirements")

    # pywebview needs a platform-specific backend extra on Linux.
    install_pywebview()


if __name__ == "__main__":
    prepare_environment()
    print("Installation Completed")