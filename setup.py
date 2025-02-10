from setuptools import setup, find_packages

setup(
    name="mimer",
    version="0.1.0",
    packages=find_packages(),  # Finds all packages inside your project
    include_package_data=True,  # Ensures data files declared in MANIFEST.in are included
    package_data={
        "": [  # Use an empty string to apply the pattern to ALL packages
            "config/**",  # Add all files inside the `config` directory
            "chroma_storage/**",  # Add all files inside `chroma_storage`
            "database/**",  # Add all files inside `database`
        ],
    },
)
