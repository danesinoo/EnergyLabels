from setuptools import setup, find_packages

setup(
    #TODO: configure with your own name
    name="ml_template",
    #TODO: configure with your own version
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'}
)