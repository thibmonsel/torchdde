import pathlib
import re

import setuptools


_here = pathlib.Path(__file__).resolve().parent

name = "torchdde"

# for simplicity we actually store the version in the __version__ attribute in the
# source
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

description = "DDEs"

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = "https://github.com/usr/" + name

license = "Apache-2.0"

author = ""
author_email = ""
python_requires = "~=3.9"

install_requires = ["torch>=2.1", "matplotlib", "numpy", "scipy", "seaborn"]

setuptools.setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    # classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    packages=setuptools.find_packages(),
)
