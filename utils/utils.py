# coding: UTF-8
"""
    @author: samuel ko
    @date:   2019.12.13
    @readme: Miscellaneous utility classes and functions.
"""

import re
import importlib
from matplotlib import pyplot as plt
import os
import sys
import types
from typing import Any, List, Tuple, Union


def plotLossCurve(opts, Loss_D_list, Loss_G_list):
    plt.figure()
    plt.plot(Loss_D_list, '-')
    plt.title("Loss curve (Discriminator)")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_discriminator.png'))

    plt.figure()
    plt.plot(Loss_G_list, '-o')
    plt.title("Loss curve (Generator)")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_generator.png'))


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    return obj.__module__ + "." + obj.__name__


def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


if __name__ == "__main__":
    def a():
        print("gaga")

    b = globals()['a']
    b = get_top_level_function_name(b)
    module,  xxx = get_module_from_obj_name(b)
    _build_func = get_obj_from_module(module, xxx)
    _build_func()
