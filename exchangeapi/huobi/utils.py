"""Module to define decorators and validators"""
from typing import List
import sys


def parse_json_to_object(json: dict, schema: object = None, **kwargs_schema: dict):
    if schema is None:
        return json
    else:
        return schema.load(json, **kwargs_schema)


def get_json_key(json: dict, json_path: str = "data"):
    if json.get('status', "") != "ok":
        if "err-msg" in json:
            raise Exception(f"{sys._getframe(1).f_code.co_name}: Wrong response status '{json.get('err-msg', None)}'.")
        else:
            raise Exception(f"{sys._getframe(1).f_code.co_name}: Wrong response status '{json.get('status', None)}'.")

    if json_path not in json:
        raise Exception(f"{sys._getframe(1).f_code.co_name}: Key {json_path} not found in response json.")

    return json[json_path]


def check_range(field_name: str, value: int, rng: List[int]):
    if value < rng[0] or value >= rng[1]:
        raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} should be in range {rng}")


def check_str(field_name: str, value: str):
    if value is None:
        raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} is required")

    if (value is not None) and (value.strip() == ""):
        raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} is required")