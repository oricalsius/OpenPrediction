"""Module to define decorators and validators"""
from marshmallow import Schema


def deserialize_json(schema: Schema, **kwargs_schema):
    def maker(func):
        async def wrapper(*args, **kwargs):
            json = await func(*args, **kwargs)
            result = schema.load(json, **kwargs_schema)

            return result
        return wrapper
    return maker
