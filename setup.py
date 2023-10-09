from setuptools import setup, find_packages

setup(
    name='alba_redcap',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'nilearn',
        'numpy',
        'openpyxl',
        'pandas',
        'pingouin',
        'plotly',
        'progressbar',
        'ptitprince',
        'scipy',
        'seaborn',
        'sklearn',
        'statsmodels',
    ],
    author='Rian Bogley',
    author_email='rianbogley@gmail.com',
    description='',
)