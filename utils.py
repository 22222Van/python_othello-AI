import os
from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Literal,
    Optional,
    Type,
    TypeVar,
    Generic,
    Callable,
    NoReturn,
    Any,
    overload,
    TYPE_CHECKING,
)
import numpy as np
from numpy.typing import NDArray

BOARD_WIDTH = 8
INF = float("inf")

CORNERS = [
    (0, 0),
    (BOARD_WIDTH - 1, 0),
    (0, BOARD_WIDTH - 1),
    (BOARD_WIDTH - 1, BOARD_WIDTH - 1),
]

star_positions = [
    (1, 0),
    (0, 1),
    (1, 1),
    (BOARD_WIDTH - 2, 0),
    (BOARD_WIDTH - 2, 1),
    (BOARD_WIDTH - 1, 1),
    (0, BOARD_WIDTH - 2),
    (1, BOARD_WIDTH - 2),
    (1, BOARD_WIDTH - 1),
    (BOARD_WIDTH - 2, BOARD_WIDTH - 2),
    (BOARD_WIDTH - 2, BOARD_WIDTH - 1),
    (BOARD_WIDTH - 1, BOARD_WIDTH - 2),
]


BLACK = 1
WHITE = -1
EMPTY = 0


PathObj = Union[str, os.PathLike]  # type: ignore
PlayerColorType = Literal[1, -1]
GridType = NDArray[np.int_]
PointType = Tuple[int, int]

T = TypeVar("T")


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
    def __get__(self, instance: None, owner: type) -> "lazy_property[T]": ...

    @overload
    def __get__(self, instance: Any, owner: type) -> T: ...

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
    mode="r",
    buffering: int = -1,
    encoding: str = "utf-8",
    *args,
    **kwargs,
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


################ COUNTER ################

import functools


def sign(x):
    """
    Returns 1 or -1 depending on the sign of x
    """
    if x >= 0:
        return 1
    else:
        return -1


class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(list(self.keys())) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = list(self.items())

        def compare(x, y):
            return sign(y[1] - x[1])

        sortedItems.sort(key=functools.cmp_to_key(compare))
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0:
            return
        for key in list(self.keys()):
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in list(y.items()):
            self[key] += value

    def __add__(self, y):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend
