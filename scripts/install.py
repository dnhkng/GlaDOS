import os
from pathlib import Path
import platform
from shutil import which
import subprocess


def is_uv_installed() -> bool:
    return which("uv") is not None


def install_uv() -> None:
    if is_uv_installed():
        print("UV is already installed")
        return

    print("Installing UV...")
    if platform.system() == "Windows":
        subprocess.run(["pip", "install", "uv"])
        subprocess.run(["uv", "self", "update"])
    else:
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)
        subprocess.run(["uv", "self", "update"])


def main() -> None:
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Install UV
    install_uv()

    # Create virtual environment
    subprocess.run(["uv", "venv", "--python", "3.12.8"])

    # Determine if CUDA is available
    if platform.system() == "Windows":
        venv_python = ".venv/bin/python"
    else:
        venv_python = ".venv/bin/python"
        os.environ["PATH"] = f"{os.path.dirname(venv_python)}:{os.environ['PATH']}"

    has_cuda = subprocess.run(["nvcc", "--version"], capture_output=True).returncode == 0
    extras = "[cuda]" if has_cuda else "[cpu]"

    # Install project in editable mode
    env = os.environ.copy()
    env["PATH"] = f"{os.path.abspath('.venv/bin')}:{env['PATH']}"
    os.environ["VIRTUAL_ENV"] = os.path.abspath(".venv")
    os.system(f"uv pip install -e .{extras}")

    # Download and verify model files
    os.system("uv run glados download")


if __name__ == "__main__":
    main()
