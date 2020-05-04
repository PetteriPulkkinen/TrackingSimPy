from setuptools import setup
from setuptools import find_namespace_packages

setup(name='trackingsimpy',
      version='0.1',
      description='Simulate different radar scenarios',
      url='https://github.com/PetteriPulkkinen/RadarSimulator',
      author='Petteri Pulkkinen',
      author_email='petteri.pulkkinen@aalto.fi',
      licence='MIT',
      packages=find_namespace_packages(),
      install_requires=[
            'numpy', 'filterpy', 'matplotlib', 'pandas', 'requests', 'bs4'
      ],
      zip_safe=False)
