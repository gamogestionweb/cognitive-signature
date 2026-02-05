from setuptools import setup, find_packages

setup(
    name="cogmind",
    version="0.1.0",
    description="CogMind: Cognitive Emulation from Cognitive Signature",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Daniel Gamo",
    author_email="",
    url="https://github.com/gamogestionweb/cognitive-signature",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "brian2": ["brian2>=2.5.0"],
        "norse": ["norse>=1.0.0"],
        "viz": ["matplotlib>=3.7.0", "plotly>=5.0.0"],
        "analysis": ["scipy>=1.10.0", "networkx>=3.0"],
        "full": [
            "brian2>=2.5.0",
            "matplotlib>=3.7.0",
            "plotly>=5.0.0",
            "scipy>=1.10.0",
            "networkx>=3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cogmind=cogmind.cogmind_runner:main",
            "cogmind-topology=cogmind.topology_generator:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
