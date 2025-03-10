from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hexcore",
    version="0.1.0",
    description="Magic: The Gathering LLM Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="John Morrissey",
    author_email="john@foundryside.dev",
    url="https://github.com/tachyon-beep/hexcore",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "psutil>=5.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mtg-loader=hexcore.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hexcore": ["data/*.json"],
    },
)
