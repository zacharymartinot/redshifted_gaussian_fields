import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="redshifted_gaussian_fields",
    version="0.0.2",
    author="Zachary Martinot",
    description="Minimally realistic simulations of redshifted Gaussian intensity fields on the sky for testing 21-cm power spectrum measurement methods.",
    long_description=long_description,
    url="https:github.com/zacharymartinot/redshifted_gaussian_fields",
    packages=setuptools.find_packages(),
    python_requires=">=3.6"
)
