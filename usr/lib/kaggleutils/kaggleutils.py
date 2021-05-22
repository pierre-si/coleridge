import subprocess, sys


def upgrade(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])


def offline_install(path, name):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-I",  # ignore (overwrite) the already installed package
            "--no-index",
            "--find-links=" + path,
            name,
        ]
    )


def offline_install_no_deps(path):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-dependencies",
            path,
        ]
    )
