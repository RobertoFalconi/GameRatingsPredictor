from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")


setup(
    name="GameRatingsPredictor",
    packages=["GameRatingsPredictor"],
    entry_points={
        "console_scripts": ['algoritmo = GameRatingsPredictor.algoritmo:main']
        },
    version="1.0.0",
    description="VideoGame ratings classifier using logistic regression, KNN and random forest.",
    long_description=long_descr,
    author="Roberto Falconi e Federico Guidi",
    url="http://www.gitlab.com/robertofalconi95/MQPI",
    install_requires=required
    )
