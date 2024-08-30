from dataclasses import dataclass, field
from typing import Any
import numpy as np

from queue import PriorityQueue as OriginalPQ


class ReservoirQueue:
    def __init__(self, accentuation_factor=30):
        """
        Queue that gets items randomly with probability accordingly to their
        weight.

        args
        ------
        accentuation_factor: float - weights are scaled to (0,1] then raised to
                                     the power of this accentuation factor.
                                     Higher accentuation factors make the queue
                                     behaviour more and more deterministically,
                                     since the difference between the highest
                                     and lowest weights weights are
                                     "accentuated."
        """
        self.items = []
        self.weights = []
        self.n = 0
        self.accentuation_factor = accentuation_factor

    def put(self, item, weight):
        if not np.isscalar(weight): 
            raise TypeError('weights must be scalar')
        if weight <= 0:
            raise ValueError(f'weights must be positive, got {weight}')
        self.items.append(item)
        self.weights.append(weight)
        self.n += 1

    def get_probabilities(self, weights):
        weights = np.array(self.weights)
        if np.any(np.isnan(weights)):
            raise ValueError(f'Weights contain NaN values: {weights}')

        s = sum(weights)
        if s == 0:
            raise ValueError('Sum of weights is zero, cannot normalize.')
        
        ps = weights / s
        # range is now [1,2], which prevents large accentuation factors driving
        # us into 0 everywhere
        ps = ps + 1
        ps = np.power(ps, self.accentuation_factor)
        
        s = sum(ps)

        ps = ps / s
        return ps

    def get(self):
        if self.n == 0:
            return None
        else:
            ps = self.get_probabilities(self.weights)

            choice_of_index = np.random.choice(
                list(range(len(self.items))), p=ps)
            choice = self.items.pop(choice_of_index)
            self.weights.pop(choice_of_index)
            self.n -= 1
            return choice

    def empty(self):
        return self.n == 0


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class PriorityQueue:
    def __init__(self):
        self.q = OriginalPQ()

    def put(self, item, weight):
        x = PrioritizedItem(priority=-weight, item=item)
        self.q.put(x)

    def get(self):
        return self.q.get().item

    @property
    def n(self) -> int:
        return self.q.qsize()

    def empty(self):
        return self.q.empty()
