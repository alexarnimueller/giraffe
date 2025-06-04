from setuptools import setup

setup(
    name="giraffe",
    version="1.2.0",
    packages=[
        "giraffe",
        "giraffe.data",
        "giraffe.configs",
        "giraffe.models",
    ],
    package_data={
        "giraffe": ["models/*/*.pt", "models/*/*.ini", "data/property_scales.json"]
    },
    url="",
    license="",
    author="Alex Mueller",
    author_email="",
    description="",
    include_package_data=True,
)
