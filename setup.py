import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bin-classifier",
    version="1.0.0",
    author="Hashem Alsaket",
    description="Automatic binary classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    include_package_data=True,
    package_data={'': ['data/*.csv']}
)