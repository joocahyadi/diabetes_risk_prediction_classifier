from setuptools import find_packages, setup
from typing import List

HYPHEN_DOT_E = '-e .'

def get_requirements(filepath:str)->List[str]:
    '''
    This function will return all the requirements
    '''
    requirements = []
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)
    
    return requirements

setup(
    name='diabetes_risk_prediction_classifier',
    version='0.0.1',
    author='joocahyadi',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)