from setuptools import setup, find_packages

setup(
    name='at_the_interstice',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'plotly',
        'kaleido',
    ],
    author='Bhuvanesh Verma',
    description='A Python package for creating interstitial plots from input dictionaries.',
    url='https://github.com/your_username/interstitial_plots_package',
    entry_points={
        'console_scripts': [
            'ati = src.pipeline:main',
        ],
    },
)
