from setuptools import setup
from os import path

# how to upload to PyPI:
# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
# briefly, first
# 1. install twice (pip install twine),
# 2. pip install importlib-metadata==4.13.0 (to avoid error
#    "AttributeError: 'EntryPoints' object has no attribute 'get'" with versions after 5.0)
#
# Then to upload to PyPI:
# 1. [bump version number in file gpac/__version__.py]
# 2. python setup.py sdist
# 3. twine upload dist/*


# this is ugly, but appears to be standard practice:
# https://stackoverflow.com/questions/17583443/what-is-the-correct-way-to-share-package-version-with-setup-py-and-the-package/17626524#17626524
def extract_version(filename: str):
    with open(filename) as f:
        lines = f.readlines()
    version_comment = '# version line; WARNING: do not remove or change this line or comment'
    for line in lines:
        if version_comment in line:
            idx = line.index(version_comment)
            line_prefix = line[:idx]
            parts = line_prefix.split('=')
            parts = [part.strip() for part in parts]
            version_str = parts[-1]
            version_str = version_str.replace('"', '')
            version_str = version_str.replace("'", '')
            version_str = version_str.strip()
            return version_str
    raise AssertionError(f'could not find version in {filename}')

package_name = 'gpac'
version = extract_version(f'{package_name}/__version__.py')
print(f'{package_name} version = {version}')


with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name=package_name,
      packages=[package_name],
      version=version,
      license='MIT',
      description=f"{package_name} stands for \"General-Purpose Analog Computer\". This package makes it easy to specify ordinary differential equations (ODEs) and view their solutions.",
      author="David Doty",
      author_email="doty@ucdavis.edu",
      url=f"https://github.com/UC-Davis-molecular-computing/{package_name}",
      long_description=long_description,
      long_description_content_type='text/markdown; variant=GFM',
      python_requires='>=3.7',
      install_requires=install_requires,
      include_package_data=True,
      )
