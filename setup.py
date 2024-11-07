from setuptools import setup, find_packages
import os

base_dir = os.path.dirname(__file__)  # Directory of the script
requirements_path = os.path.join(base_dir, 'requirements.txt')

with open(requirements_path) as f:
    required = f.read().splitlines()
      
setup(name='treeQuadrature',
      version='0.1',
      description='Space partition trees for high dimensional integration',
      url='http://github.com/thomfoster',
      author='Thomas Foster',
      author_email='thomas.foster@keble.ox.ac.uk',
      license='MIT',
      packages=find_packages(),
      install_requires=required,
      include_package_data=True,
      zip_safe=False)
