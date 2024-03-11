
# ==============================================================================
import os

# from codecs import open

from setuptools import setup, find_packages


setup(
    name='swincell',
    version='0.1',
    packages=find_packages(),
    description='A transformer based cell segmentation framework',
    long_description=open('README.md').read(),
    long_description_content_type='***',
    author='***',
    author_email='****@gmail.com',
    url='https://github.com/xzhang0123/swincell',
    install_requires=[
        'scikit-image>=0.19.3',
        'tqdm>=4.64.0',
        'monai>=0.10',
        'torch>=1.8.1',
        'torchvision>=0.9.1',
        'libtiff>=0.4.2',
        'numpy>=1.22.3',
        'seaborn>=0.12.1',
        #
        
        
        'scipy>=1.5.4',
        #'scikit-learn>=1.1.3',
        'matplotlib>=3.5.2',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    # ]
)
