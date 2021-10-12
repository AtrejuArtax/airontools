import setuptools
import os

PACKAGE_NAME = 'airontools'

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name=PACKAGE_NAME,
    version='0.1.8',
    scripts=[],
    author='Claudi Ruiz Camps',
    author_email='claudi_ruiz@hotmail.com',
    description='Machine learning tools to complement the AIronSuit package.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AtrejuArtax/airontools',
    packages=setuptools.find_packages(
        include=[PACKAGE_NAME] + [PACKAGE_NAME + '.' + name
                                  for name in os.listdir(os.path.join(os.getcwd(), PACKAGE_NAME))
                                  if not any([str_ in name for str_ in ['.py', '__']])]),
    install_requires=['sklearn'],
    classifiers=['Programming Language :: Python :: 3',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent'],
    license='BSD')
