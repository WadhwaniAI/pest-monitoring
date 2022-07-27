from __future__ import annotations  # Remove once we move to 3.10

import itertools as it
import functools as ft
import collections as cl

BatchInfo = cl.namedtuple(
    "BatchInfo",
    "locations, classes, labels",
    defaults=[None, None, None],
)


class LossValue(cl.namedtuple("LossValue", "name, loss, weight", defaults=[1.0])):
    """Representation of value calculed by a loss function

    Parameters
    ----------
    name: str
        Name of the loss calculator responsible for creating the type
    loss: float, Tensor
        Value of the loss
    weight: float, optional
        Weight applied to this loss
    """

    __slots__ = ()

    def __str__(self):
        """

        Returns
        -------
        str
            Name of the loss calculator responsible for creating the type
        """

        return self.name


class _LossTrail:
    """Abstract class for loss value collection

    Attributes
    ----------
    trail: list
        Collection of LossValues

    Notes
    -----
    This class should not be instantiated directly. Instead use `LossTrail`.

    """

    def __init__(self):
        self.trail = []

    def __str__(self):
        """

        Returns
        -------
        str
            Comma separated list of LossValue::name's held in the trail
        """

        return ",".join(map(str, self))

    def __iter__(self):
        """

        Yields
        ------
        LossValue
            LossValue's held in the trail
        """

        yield from self.trail

    def __getitem__(self, key):
        """

        Parameters
        ----------
        key: str
            Name of loss value desired

        Yields
        ------
        LossValue
            LossValue's with whose name matches the key
        """

        yield from filter(lambda x: str(x) == key, self)

    def __add__(self, other):
        """

        Parameters
        ----------
        other: LossValue, LossTrail

        Returns
        -------
        LossTrail
            A new trail with the value added, or the trail extended
        """

        raise NotImplementedError()

    def to_records(self):
        """

        Yields
        ------
        dict
            LossValue's as dictionaries
        """

        yield from (x._asdict() for x in self)

    #
    # These properties exist, and are strategically named, to keep
    # type access consistent with LossValue
    #

    @property
    def loss(self):
        """

        Returns
        -------
        float, Tensor
            Sum of loss values in this trail multiplied by their corresponding
            weight
        """

        return sum(x.loss * x.weight for x in self)

    @property
    def name(self):
        """

        Returns
        -------
        str
            see `__str__`
        """

        str(self)

    @property
    def weight(self):
        """

        Returns
        -------
        float
            Sum of weights in the trail
        """

        return sum(x.weight for x in self)


class LossTrail(_LossTrail):
    """Concrete LossTrail

    This class should be instantiated by clients.

    Notes
    -----
    This class exists to facilitate addition between two loss
    trails. Addition relies on Python's single dispatch functionality,
    which cannot handle recursive types. See source for more details.
    """

    @ft.singledispatchmethod
    def __add__(self, other):
        """
        Concrete implementation of the parents abstraction.

        Parameters
        ----------
        other: LossValue, LossTrail

        Returns
        -------
        LossTrail
            A new trail with the value added, or the trail extended
        """

        raise TypeError()

    @__add__.register
    def _(self, other: LossValue):
        """
        Adds a LossValue to the underlying container of LossValue's.

        Parameters
        ----------
        other: LossValue

        Returns
        -------
        LossTrail
            The same instance that called method
        """

        self.trail.append(other)
        return self

    @__add__.register
    def _(self, other: _LossTrail):
        """
        Creates a new LossTrail containing the combination of
        trails from the caller and the parameter.

        Parameters
        ----------
        other: LossTrail

        Returns
        -------
        LossTrail
            A new LossTrail instance
        """

        cls = type(self)()
        cls.trail.extend(it.chain(self, other))
        return cls
