from setuptools import setup
from setuptools import find_namespace_packages

setup(name='radarsim',
      version='0.1',
      description='Simulate different radar scenarios',
      url='https://github.com/PetteriPulkkinen/RadarSimulator',
      author='Petteri Pulkkinen',
      author_email='petteri.pulkkinen@aalto.fi',
      licence='MIT',
      packages=find_namespace_packages(),
      install_requires=[
            'numpy', 'filterpy', 'matplotlib'
      ],
      zip_safe=False)
