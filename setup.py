import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='airontools',
    version='0.1.0',
    scripts=[],
    author='Claudi Ruiz Camps',
    author_email='claudi_ruiz@hotmail.com',
    description='Tools to complement the AIronSuit package.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AtrejuArtax/airontools',
    packages=setuptools.find_packages(),
    classifiers=['Programming Language :: Python :: 3',
                 'License :: BSD 3-Clause',
                 'Operating System :: OS Independent'])
