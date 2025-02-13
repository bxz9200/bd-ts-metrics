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
    url='https://github.com/bxz9200/bd-ts-metrics',
    license='MIT',
    install_requires=['numpy>=1.19', 'matplotlib>=3.9', 'scipy>=1.13', 'dtaidistance==2.3.12', 'tensorflow>=2.15', 'scikit-learn>=1.5',
                      'torch>=2.3', 'statsmodels>=0.13', 'tslearn==0.6.3', 'seaborn>=0.12.2', 'mgzip==0.2.1', 'protobuf==4.25.5','Jinja2==3.1.5', 'GitPython>=3.1'],

    classifiers=["Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Programming Language :: Python :: 3.10"]
)