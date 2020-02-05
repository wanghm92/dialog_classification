from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['spacy>=2.2.3', 'tensorflow-datasets>=2.0.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
