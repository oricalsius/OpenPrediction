"""Module to define decorators and validators"""
from typing import List, Any, Union
from pandas import core, DataFrame
from jsonpath_ng import parse
import sys


def parse_json_to_object(json: Union[dict, List], schema: Any = None, **kwargs_schema: dict):
    if schema is None:
        return json
    elif schema is core.frame.DataFrame:
        return DataFrame(json)
    else:
        key = getattr(schema, "global_schema_key", None)
        if key is not None and key.strip() != '':
            return schema.load({key: json}, **kwargs_schema)
        else:
            return schema.load(json, **kwargs_schema)


def get_json_key(json: dict, json_path: str = "data"):
    if json.get('status', "") != "ok" and json.get('Response', "") != "Success":
        if "err-msg" in json:
            raise Exception(f"{sys._getframe(1).f_code.co_name}: Wrong response status '{json.get('err-msg', None)}'.")
        else:
            raise Exception(f"{sys._getframe(1).f_code.co_name}: Wrong response status '{json.get('status', None)}'.")

    exp = parse(json_path)
    res = exp.find(json)
    if not res:
        raise Exception(f"{sys._getframe(1).f_code.co_name}: Key {json_path} not found in response json.")

    return res.pop().value


def check_range(field_name: str, value: int, rng: List[int]):
    if value < rng[0] or value >= rng[1]:
        raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} should be in range {rng}")


def check_str(field_name: str, value: str):
    if value is None:
        raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} is required")

    if (value is not None) and (value.strip() == ""):
        raise Exception(f"{sys._getframe(1).f_code.co_name}: parameter {field_name} is required")