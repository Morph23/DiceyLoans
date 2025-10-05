"""Setup configuration for DiCE package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dice-counterfactuals",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="DiCE: Diverse Counterfactual Explanations for Tabular Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dice-counterfactuals",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    keywords="counterfactuals, explainable-ai, machine-learning, interpretability, fairness",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/dice-counterfactuals/issues",
        "Source": "https://github.com/yourusername/dice-counterfactuals",
        "Documentation": "https://github.com/yourusername/dice-counterfactuals/wiki",
    },
)