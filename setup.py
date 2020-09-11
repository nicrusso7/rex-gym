import os
import platform
from distutils.core import setup

from setuptools import find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
os_name = platform.system()

with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read()


def copy_assets(dir_path):
    base_dir = os.path.join('rex_gym', dir_path)
    if os_name == 'Windows':
        sep = '\\'
    else:
        sep = '/'
    for (dirpath, dirnames, files) in os.walk(base_dir):
        for f in files:
            yield os.path.join(dirpath.split(sep, 1)[1], f)


setup(
    name='rex_gym',
    version='0.2.1',
    license='Apache 2.0',
    packages=find_packages(),
    author='Nicola Russo',
    author_email='dott.nicolarusso@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nicrusso7/rex-gym',
    download_url='https://github.com/nicrusso7/rex-gym/archive/master.zip',
    install_requires=requirements,
    entry_points='''
        [console_scripts]
        rex-gym=rex_gym.cli.entry_point:cli
    ''',
    package_data={
        '': [f for f in copy_assets('policies')] + [a for a in copy_assets('util')]
    },
    keywords=['openai', 'gym', 'robot', 'quadruped', 'pybullet', 'ai', 'reinforcement learning', 'machine learning',
              'RL', 'ML', 'tensorflow', 'spotmicro', 'rex'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Framework :: Robot Framework :: Library',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7']
)
