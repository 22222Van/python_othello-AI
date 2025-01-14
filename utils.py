import os
from enum import Enum
from typing import (
    List, Tuple, Union, Literal, Optional, Type, TypeVar, Generic, Callable,
    NoReturn, Any, overload, TYPE_CHECKING
)

BOARD_WIDTH = 8
INF = float('inf')


class PieceColor(Enum):
    BLACK = 1
    WHITE = -1
    EMPTY = 0


BLACK = PieceColor.BLACK
WHITE = PieceColor.WHITE
EMPTY = PieceColor.EMPTY


PathObj = Union[str, os.PathLike[str]]
PlayerColorType = Literal[PieceColor.BLACK, PieceColor.WHITE]  # FIXME
GridType = List[List[PieceColor]]
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

# 这个函数目前代码里没有用，但是保存模型时可能需要使用，先复制过来了


def safe_open(
    filepath: PathObj,
    mode='r',
    buffering: int = -1,
    encoding: str = 'utf-8',
    *args,
    **kwargs
):
    """
    Opens a file safely, creating the parent directory if it does not exist.
    Also sets the default encoding to 'utf-8', unlike the built-in `open`
    function.

    Note: This function may still throw an error, for example
    - OSError: If the file cannot be opened for any reason other than missing
      parent directories.
    - FileNotFoundError: If the specified path is invalid or cannot be created.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    return open(filepath, mode, buffering, encoding, *args, **kwargs)
