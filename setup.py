from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

setup(name='nytorch',
      version="1.0",
      author='jimmyzzzz',
      author_email='<sciencestudyjimmy@gmail.com>',
      description='Nytorch enhances PyTorch with advanced particle operations, seamlessly integrating new functionalities for effortless compatibility and enrichment.',
      long_description_content_type="text/markdown",
      long_description=long_description,
      packages=find_packages(),
      install_requires=['torch>=1.7'],
      python_requires='>=3.8',
      url='https://github.com/jimmyzzzz/nytorch',
      license='BSD 2-Clause License',
      classifiers=['License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 3',
                   'Operating System :: OS Independent'])
