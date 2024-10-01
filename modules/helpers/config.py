import importlib


def load_config(name):
    return importlib.import_module(f"config.{name}")
