import numpy as np

from queue import PriorityQueue as OriginalPQ

class ReservoirQueue:
    def __init__(self, accentuation_factor=30):
        """
        Queue that gets items randomly with probability accordingly to their weight.
        
        args
        ------
        accentuation_factor: float - weights are scaled to (0,1] then raised to the power of 
                                     this accentuation factor. Higher accentuation factors make
                                     the queue behaviour more and more deterministically, since 
                                     the difference between the highest and lowest weights weights
                                     are "accentuated."
        """
        self.items = []
        self.weights = []
        self.n = 0
        self.accentuation_factor = accentuation_factor

    def put(self, item, weight):
        assert np.isscalar(weight), 'weights must be scalar'
        assert weight >= 0, 'weights must be >= 0'
        self.items.append(item)
        self.weights.append(weight)
        self.n += 1

    def get(self):
        if self.n == 0:
            return None
        else:
            weights = np.array(self.weights)
            s = sum(weights)
            ps = weights / s
            ps = ps + 1   # range is now [1,2], which prevents large accentuation factors driving us into 0 everywhere
            ps = np.power(ps, self.accentuation_factor)
            s = sum(ps)
            ps = ps / s

            choice_of_index = np.random.choice(list(range(len(self.items))), p=ps)
            choice = self.items.pop(choice_of_index)
            self.weights.pop(choice_of_index)
            self.n -= 1
            return choice
    
    def empty(self):
        return self.n == 0

from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

class PriorityQueue:
    def __init__(self):
        self.q = OriginalPQ()
    
    def put(self, item, weight):
        x = PrioritizedItem(priority=-weight, item=item)
        self.q.put(x)

    def get(self):
        return self.q.get().item

    def empty(self):
        return self.q.empty()