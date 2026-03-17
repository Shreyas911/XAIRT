from setuptools import setup, find_packages

setup(
    name="XAIRT",
    version="1.0.0",
    author="Your Name",
    description="A PyTorch-based eXplainable AI Regression Toolkit",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "captum",  # The core PyTorch XAI engine replacing innvestigate
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
