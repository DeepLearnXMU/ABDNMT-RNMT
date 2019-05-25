import os
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python 3.7 is required for thseq.')

with open('README.md') as r:
    readme = r.read(),
with open('LICENSE') as r:
    license = r.read()
with open('requirements.txt') as r:
    req = r.read()
setup(
    name='thseq',
    version='0.1.0dev',
    description='Sequence-to-Sequence Toolkit Based on PyTorch',
    author='Xiangwen Zhang',
    author_email='xwzhang@stu.xmu.edu.cn',
    long_description=readme,
    license=license,
    install_requires=req.strip().split(os.linesep),
    packages=find_packages(),
)
