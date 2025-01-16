import os
from typing import (
    List, Dict, Tuple, Union, Literal, Optional, Type, TypeVar, Generic, Callable,
    NoReturn, Any, overload, TYPE_CHECKING
)
import numpy as np
from numpy.typing import NDArray

BOARD_WIDTH = 8
INF = float('inf')

star_positions = [
    (1, 0),
    (0, 1),
    (1, 1),
    (BOARD_WIDTH-2, 0),
    (BOARD_WIDTH-2, 1),
    (BOARD_WIDTH-1, 1),
    (0, BOARD_WIDTH-2),
    (1, BOARD_WIDTH-2),
    (1, BOARD_WIDTH-1),
    (BOARD_WIDTH-2, BOARD_WIDTH-2),
    (BOARD_WIDTH-2, BOARD_WIDTH-1),
    (BOARD_WIDTH-1, BOARD_WIDTH-2)
]


BLACK = 1
WHITE = -1
EMPTY = 0


PathObj = Union[str, os.PathLike]  # type: ignore
PlayerColorType = Literal[1, -1]
GridType = NDArray[np.int_]
PointType = Tuple[int, int]

T = TypeVar('T')


class lazy_property(Generic[T]):
    """
    A decorator that transforms a method into a lazily evaluated property.

    The property is computed and cached upon its first access. Subsequent
    accesses return the cached value without re-computation.

    Example:
    --------

    >>> class Example:
    ...     def __init__(self, x):
    ...         self.x = x
    ...
    ...     @lazy_property
    ...     def expensive_computation(self):
    ...         print("Performing expensive computation...")
    ...         time.sleep(1)
    ...         return self.x ** 2
    ...
    >>> obj = Example(10)

    >>> print(obj.expensive_computation)
    Performing expensive computation...
    100

    >>> print(obj.expensive_computation)
    100

    Note:
    --------

    - The decorated method must accept only the instance (`self`) as its
      parameter.
    - Cached elements are read-only and cannot be assigned a value.
    """

    def __init__(self, func: Callable[[Any], T]) -> None:
        super().__init__()
        self.func = func
        self.attr_name = f"__lazy_{func.__name__}_{func.__qualname__.split('.')[0]}"

    @overload
    def __get__(self, instance: None, owner: type) -> "lazy_property[T]":
        ...

    @overload
    def __get__(self, instance: Any, owner: type) -> T:
        ...

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if not hasattr(instance, self.attr_name):
            setattr(instance, self.attr_name, self.func(instance))
        return getattr(instance, self.attr_name)

    def __set__(self, instance, value) -> NoReturn:
        raise AttributeError(
            f"Cannot assign to a read-only property '{self.func.__name__}'."
        )
