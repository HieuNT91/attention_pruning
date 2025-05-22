from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as f:
        lines = f.read().splitlines()
        # Skip comments and blank lines
        reqs = [line for line in lines if line and not line.startswith("#")]
    return reqs

setup(
    name="datadistillation",
    version="0.1",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
)