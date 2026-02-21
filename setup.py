from setuptools import setup, find_packages
setup(
    name='AIvsHuman',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        # "numpy==1.26.4",   # if you must pin for Colab compatibility
        "pyDOE",
        "pyDOE2",
        # "scipy",
        # "matplotlib",
        # "ipywidgets",
        # "ipython",
    ],
)
