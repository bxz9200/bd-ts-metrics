import setuptools

setuptools.setup(
    name="bdtsmetrics",
    version="0.1.0",
    author="Bingyin Zhao",
    author_email="bingyin@betterdata.ai",
    description="Betterdata time series metrics",
    long_description="Betterdata time series metrics long description",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url='https://github.com/bxz9200/bd-ts-metrics.git',
    license='MIT',
    install_requires=['numpy==1.18.5', 'matplotlib==3.5.3', 'scipy>=1.4', 'dtaidistance==2.3.12', 'tensorflow==1.15.5', 'scikit-learn==1.0.2',
                      'torch==1.13.1', 'statsmodels==0.13.5', 'tslearn==0.6.3', 'seaborn==0.12.2', 'mgzip==0.2.1', 'protobuf==3.20.0'],

    classifiers=["Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Programming Language :: Python :: 3.10"]
)