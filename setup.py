from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(name='auto-proof', 
        version='1.0',
        description="Automated Proofreading for Connectomics",
        author="Shawn Stanley",
        author_email="shawns28@uw.edu",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3"
                 ],
        packages=find_packages(),
        install_requires=required,
        python_requires=">=3.9",
        )