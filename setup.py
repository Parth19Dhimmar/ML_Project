from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path : str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
name = "mlproject",
version = "0.0.1",
author = "Parth",
author_email = "parth1211dhimmar@gmail.com",
packges = find_packages(),                     #directories with __init__ are taken as pakage
install_requires = get_requirements('requirements.txt')
)