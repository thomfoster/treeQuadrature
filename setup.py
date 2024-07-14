from setuptools import setup, find_packages

setup(name='treeQuadrature',
      version='0.1',
      description='Space partition trees for high dimensional integration',
      url='http://github.com/thomfoster',
      author='Thomas Foster',
      author_email='thomas.foster@keble.ox.ac.uk',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)