from setuptools import setup, find_packages

setup(
    name='graph_utils',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'networkx==2.7',
        'pandas',
        'numpy',
        'scikit-learn',
        'optuna',
        'tqdm',
        'igraph',
        'xgboost',
    ],
    author='Zahiriddin Rustamov',
    author_email='zahir@uaeu.ac.ae',
    description='Necessary utility functions for performing graph reduction experiment.',
)