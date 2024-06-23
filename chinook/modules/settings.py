"""
settings.py

This module defines functions and variables related to environmental settings and configuration for the chatbot
agent. It includes functions for retrieving environment variables and constants for various settings.
"""

import os
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).parent.resolve()


def get_env_var(var_name: str, cast_as: Callable = str, default=None, as_list: bool = False) -> Any:
    """
    Retrieves an environment variable and casts it to a specified type.

    Parameters:
    -----------
    var_name: str
        The name of the environment variable to retrieve.
    cast_as: Callable
        The type to cast the environment variable to.
    default: Any
        The default value to return if the environment variable is not found.
    as_list: bool
        Whether to return the value as a list.
    """
    try:
        value = os.environ[var_name]  # if variable is found
    except KeyError as e:  # if variable isnt found
        if default is not None:
            return default  # return the default value if it has been provided
        print(
            f"""Environment variable {var_name} is not set
        and no default value was provided.\n Error: {e}"""
        )
        raise KeyError  # if default value hasnt been provided and variable isnt found raise error

    try:
        if "," in value:  # if the value from os.environ has a comma in it
            return [cast_as(v) for v in value.split(",")]  # return the values in a list
        if as_list:  # only triggers if value from os.environ doesnt have a comma in it
            return [cast_as(value)]  # return the value in a list
        return cast_as(value)  # if no comma, and as_list is False
    except ValueError as e:
        print(
            f"""Environment variable {var_name} with value {value}
        could not be cast to {cast_as}.\n Error: {e}"""
        )
        raise ValueError


# MSSQL Database
JUPYTER_PORT_APP = get_env_var("JUPYTER_PORT_APP", cast_as=int)
POSTGRES_USER=get_env_var("POSTGRES_USER", cast_as=str)
POSTGRES_PASSWORD=get_env_var("POSTGRES_PASSWORD", cast_as=str)
POSTGRES_DB=get_env_var("POSTGRES_DB", cast_as=str)