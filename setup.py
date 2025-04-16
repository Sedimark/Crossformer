from setuptools import setup, find_packages

setup(
    name="crossformer-pruned",
    version="__VERSION__",
    author="Sneha H, Dr. Peipei Wu (Paul)",
    author_email="sneha.rmh@gmail.com, peipeiwu1996@gmail.com",
    description="A Pruned version of CrossFormer for lightweight multivariate time series forecasting",
    long_description=open("README_package.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sedimark/Surrey_AI/tree/feature_pruning",
    license="EUPL-1.2",
    packages=find_packages(
        include=["crossformer", "crossformer.*"],
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=["torch", "lightning", "einops", "pandas"],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    include_package_data=True,
)
