"""Module to define decorators and validators"""
from typing import List
from marshmallow import Schema


def get_json_key(json: dict, json_path: str = "data"):
    if json.get('status', "") != "ok":
        raise Exception(f"Wrong response status '{json.get('status', None)}'.")

    if json_path not in json:
        raise Exception("key data not found in response json.")

    return json[json_path]


def check_range(field_name: str, value: int, rng: List[int]):
    if value < rng[0] or value >= rng[1]:
        raise Exception(f"{field_name} parameter should be in range {rng}")


def check_str(field_name: str, value: str):
    if value is None:
        raise Exception(f"{field_name} parameter is required")

    if (value is not None) and (value.strip() == ""):
        raise Exception(f"{field_name} parameter is required")