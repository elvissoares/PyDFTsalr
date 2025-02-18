import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydftsalr",
    version="0.0.1",
    author="Elvis Soares", #<<<
    author_email="elvis@peq.coppe.ufrj.br", #<<<
    description="A DFT package for SALR fluids in Python", #<<<
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elvissoares/PyDFTsalr",  #<<<
    install_requires=[         
        'pandas',         
        'numpy',
        'matplotlib',
        'scipy',
        'scienceplots',
        'torch',
    ],
    packages=setuptools.find_packages(
        where='.',
        include=['pydftsalr*'],  # alternatively: `exclude=['additional*']`
        ),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "Operating System :: OS Independent",
    ],
)
