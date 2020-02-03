from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.rst')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

version = {}
with open(os.path.join(_here, 'somepackage', 'version.py')) as f:
    exec(f.read(), version)

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pbt',
    version=version['__version__'],
    description=('Finding a good hyper-parameter schedule for neural networks.'),
    long_description="PBT is a novel, Lamarckian evolutionary approach to hyper-parameter optimization for selecting the optimal hyper-parameter configuration and machine learning model by training a series of neural network models in parallel. The method can be performed as quickly as other methods and has shown to outperform random search in model performance on various benchmarks in deep reinforcement learning using A3Cstyle methods, as well as in supervised learning for machine translation and Generative Adversarial Networks (GANs). While similar procedures have been explored independently, PBT has gained increasing amount of attention since it was proposed. There has already been seen shown various use cases of PBT in AutoML, e.g. packages for HPO tuning and frameworks. PBT have also streamlined the experiment testing in different application-based domains with different machine learning approaches such as auto-encoders, reinforcement learners, neural networks and generative adversarial networks.",
    author='Thomas Angeland',
    author_email='thomas.angeland@gmail.com',
    url='https://github.com/alvtron/PopulationBasedTraining',
    license='LICENSE',
    packages=['somepackage'],
#   no dependencies in this example
#   install_requires=[
#       'dependency==1.2.3',
#   ],
#   no scripts in this example
#   scripts=['bin/a-script'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6'],
    )