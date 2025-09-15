from setuptools import setup, find_packages

setup(
    name='scLinguist',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='MIT',
    author='Min Li Lab',
    author_email='zy_fang@csu.edu.cn',
    description='a Foundation Model for Cross-Modality Translation in Single-Cell Omics',
    package_data={
        'scLinguist': ['data/gene_order.csv']
    },
)