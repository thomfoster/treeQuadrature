import numpy as np

class ReservoirQueue:
    def __init__(self, accentuation_factor=1):
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
        self.items.append(item)
        self.weights.append(weight)
        self.n += 1

    def get(self):
        if self.n == 0:
            return None
        else:
            min_val = min(self.weights)
            range_val = 1 + max(self.weights) - min_val
            f = lambda w : ((w - min_val + 1e-5)/range_val)**self.accentuation_factor
            probabilities = [f(weight) for weight in self.weights]
            sum_weights = sum(probabilities)
            probabilities = [p / sum_weights for p in probabilities]
            choice_of_index = np.random.choice(list(range(len(self.items))), p=probabilities)
            choice = self.items.pop(choice_of_index)
            chosen_weight = self.weights.pop(choice_of_index)
            self.n -= 1
            return choice
    
    def empty(self):
        return self.n == 0
