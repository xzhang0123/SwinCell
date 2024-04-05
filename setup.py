
import os

# from codecs import open

from setuptools import setup, find_packages

setup(
    name='swincell',
    version='0.6',
    packages=find_packages(),
    description='A transformer based cell segmentation framework',
    long_description=open('README.md').read(),
    long_description_content_type='***',
    author='***',
    author_email='****@gmail.com',
    url='https://github.com/xzhang0123/swincell',
    install_requires=[
        'numpy>=1.21.5',
        'scikit-image>=0.19.3',
        'tqdm>=4.64.0',
        'monai[all]>=0.10',
        'torch>=1.8.1',
        
        'libtiff>=0.4.2',
        
        'seaborn>=0.12.1',
        
        #
        'scipy>=1.7.1',
        #'scikit-learn>=1.1.3',
        'matplotlib>=3.8.3',
        'pandas==1.3.4',
        'opencv_python_headless==3.4.18.65',
        'tifffile==2022.8.12',
        'imagecodecs==2024.1.1',
        'tracker==0.1.1',
        'natsort==8.3.1',
        'tensorboardX==2.6.2.2', # version TBD
        'numba==0.59.0',  # version TBD
        'csbdeep==0.7.4',
        'nibabel>=4.0.1',
        #from config
        'pytorch-ignite>=0.4.8',
        'torchvision>=0.12.0',
        'transformers>=4.18.0',
        'mlflow>=1.26.1',
        'gdown>=4.4.0',
        'fastremap==1.13.4',


        # additional packages
        'pyyaml',


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
