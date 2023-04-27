#########################################################################################
# The code in this file has been taken & modified from the homeworks of the course IFT6163 at UdeM.
# Original authors: Glen Berseth, Lucas Maes, Aton Kamanda
# Date: 2023-04-06
# Title: ift6163_homeworks_2023
# Code version: 5a7e39e78a9260e078555305e669ebcb93ef6e6c
# Type: Source code
# URL: https://github.com/milarobotlearningcourse/ift6163_homeworks_2023
#########################################################################################

import inspect
from functools import wraps


def hidden_member_initialize(wrapped__init__):
    """Decorator to initialize members of a class with the named arguments. (i.e. so D.R.Y. principle is maintained
    for class initialization).

    Modified from http://stackoverflow.com/questions/1389180/python-automatically-initialize-instance-variables
    :param wrapped__init__:
    :returns:
    :rtype:

    """

    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, "_" + name, arg)

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, "_" + names[index]):
                    setattr(self, "_" + names[index], defaults[index])

        wrapped__init__(self, *args, **kargs)

    return wrapper
