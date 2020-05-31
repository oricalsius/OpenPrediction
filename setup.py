from setuptools import setup

setup(
    name="price-prediction-api",
    version="0.0.1",
    packages=["api", "database", "exchangeapi", "exchangeapi.huobi", "exchangeapi.huobi.models",
              "exchangeapi.huobi.unittests", "strategies", "strategies.indicators"],

    install_requires=['requests', 'aiohttp', 'asyncio', 'pandas', 'numpy', 'marshmallow', 'sqlalchemy', 'plotly',
                      'joblib', 'scikit-learn', 'scipy', 'numba', 'keras-tuner', 'tensorflow', 'keras', 'pymongo']
)