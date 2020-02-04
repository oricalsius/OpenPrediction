from setuptools import setup

setup(
    name="price-prediction-api",
    version="0.0.1",
    packages=["exchangeapi", "exchangeapi.huobi", "exchangeapi.huobi.models", "exchangeapi.huobi.unittests",
              "api", "database"],
    install_requires=['requests', 'aiohttp', 'asyncio', 'pandas', 'numpy', 'marshmallow', 'sqlalchemy']
)