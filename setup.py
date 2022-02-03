"""
Copyright 2021 Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from distutils.core import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name='genedisco',
    version='1.0.3',
    python_requires=">=3.8",
    packages=find_packages(),
    package_data={
        "": ["*.txt"],
        "": ["*.csv"],
    },
    author='see README.txt',
    url="https://gsk.ai/genedisco-challenge/",
    author_email='patrick.x.schwab@gsk.com',
    license="Apache-2.0",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'run_experiments=genedisco.apps.run_experiments_application:main',
            'active_learning_loop=genedisco.apps.active_learning_loop:main'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
