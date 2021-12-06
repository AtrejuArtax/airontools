import setuptools

PACKAGE_NAME = 'airontools'
SUB_PACKAGES_NAMES = [
    'airontools.constructors',
    'airontools.constructors.layers',
    'airontools.constructors.models',
    'airontools.constructors.models.supervised',
    'airontools.constructors.models.supervised.classification',
    'airontools.constructors.models.unsupervised',
    'airontools.constructors.utils',
]

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name=PACKAGE_NAME,
    version='0.1.14',
    scripts=[],
    author='Claudi Ruiz Camps',
    author_email='claudi_ruiz@hotmail.com',
    description='Machine learning tools to complement the AIronSuit package.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AtrejuArtax/airontools',
    packages=setuptools.find_packages(
        include=[PACKAGE_NAME] + SUB_PACKAGES_NAMES),
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
        'tensorflow==2.7.0',
        'tensorboard==2.7.0'
    ],
    classifiers=['Programming Language :: Python :: 3',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent'],
    license='BSD')
