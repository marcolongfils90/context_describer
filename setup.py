"""Setup file."""
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "1.0.0"

PROJECT_NAME = "context_describer"
AUTHOR_NAME = "marcolongfils90"
AUTHOR_EMAIL = "marcolongfils@gmail.com"
SRC_REPO = "context_describer"

setuptools.setup(
    name=PROJECT_NAME,
    version=__version__,
    description=("Simple image detection and context describer."),
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    long_description=long_description,
    long_description_content="text/markdown",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires='>=3.6')
