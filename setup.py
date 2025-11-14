"""
Setup script for Ensemble-Redaction Consensus Pipeline
"""

from setuptools import setup, find_packages

with open("README_NEW.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ensemble-privacy-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Training-Free Privacy-Preserving LLM Pipeline for Sensitive User Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ensemble-privacy-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "datasets>=2.14.0",
        "huggingface_hub>=0.17.0",
    ],
    extras_require={
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "google-generativeai>=0.3.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "ensemble-privacy=src.pipeline:main",
        ],
    },
    include_package_data=True,
    keywords="privacy llm ensemble differential-privacy pii gdpr",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ensemble-privacy-pipeline/issues",
        "Source": "https://github.com/yourusername/ensemble-privacy-pipeline",
        "Documentation": "https://github.com/yourusername/ensemble-privacy-pipeline/tree/main/docs",
    },
)
