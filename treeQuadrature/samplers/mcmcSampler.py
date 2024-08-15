from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any, Tuple

from .sampler import Sampler

class Proposal(ABC):
    @abstractmethod
    def propose(self, current_sample: np.ndarray, 
                lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
        """
        Generate a new sample based on the current sample.
        
        Parameters
        ----------
        current_sample : np.ndarray
            The current sample.
        lows, highs: np.ndarray
            the lower and upper boundaries of sampling region
        
        Returns
        -------
        np.ndarray
            The proposed new sample.
            same shape as current_sample
        """
        pass

    @abstractmethod
    def density(self, new_sample: np.ndarray, 
                current_sample: np.ndarray) -> float:
        """
        Calculate the proposal density of transitioning 
        from current_sample to new_sample.
        
        Parameters
        ----------
        new_sample : np.ndarray
            The new proposed sample.
        current_sample : np.ndarray
            The current sample.
        
        Returns
        -------
        float
            The density of proposing `new_sample` from `current_sample`.
        """
        pass

class GaussianProposal(Proposal):
    """
    Gaussian proposal which avoids generating samples at the boundaries 
    """
    def __init__(self, std: float):
        self.std = std

    def propose(self, current_sample: np.ndarray, 
                lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
        proposal = current_sample + np.random.normal(
            0, self.std, size=current_sample.shape
            )
        # Clip to avoid boundary issues
        proposal = np.clip(proposal, lows + 1e-6, 
                           highs - 1e-6)
        return proposal

    def density(self, new_sample: np.ndarray, 
                current_sample: np.ndarray) -> float:
        # factor outside exponent will be cancelled in acceptance ratio
        exponent = -np.sum((new_sample - current_sample) ** 2
                           ) / (2 * self.std ** 2)
        return np.exp(exponent)
    
class McmcSampler(Sampler):
    """
    MCMC sampler that generates samples from 
    the modulus of f
    using the Metropolis-Hastings algorithm.
    """

    def __init__(self, proposal: Optional[Proposal]=None,
                 burning: int=100):
        """
        Arguments
        ---------
        proposal : Proposal, Optional
            Default: GaussianProposal(std=0.5)
        burning : int, Optional
            Number of initial samples to discard
            Defaults to 100
        """
        if proposal is not None:
            self.proposal = proposal
        else:
            self.proposal = GaussianProposal(std=0.5)

        self.burning = burning

    def rvs(self, n: int, mins: np.ndarray, maxs: np.ndarray,
            f: callable,
            **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate MCMC samples.

        Parameters
        ----------
        n : int 
            Number of samples.
        mins, maxs : np.ndarray
            1 dimensional arrays of the lower bounds
            and upper bounds
        f : function
            the integrand

        Returns
        -------
        np.ndarray
            Samples from the modulus of the integrand.
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {n}")
        mins, maxs, D = Sampler.handle_mins_maxs(mins, maxs)

        xs = np.zeros((n+self.burning, D))
        ys = np.zeros((n+self.burning))
        current_sample = np.random.uniform(low=mins, 
                                           high=maxs, size=D)
        
        for i in range(n+self.burning):
            proposal = self.proposal.propose(current_sample, 
                                             mins, maxs)
            proposal = np.clip(proposal, mins, maxs)
            
            current_value = np.abs(f(current_sample.reshape(1, -1)))[0]
            proposal_value = np.abs(f(proposal.reshape(1, -1)))[0]

            proposal_density_forward = self.proposal.density(proposal, current_sample)
            proposal_density_backward = self.proposal.density(current_sample, proposal)

            acceptance_ratio = min(1, (proposal_value / current_value) * 
                                   (proposal_density_backward / proposal_density_forward))
            
            if np.random.rand() < acceptance_ratio:
                current_sample = proposal
                current_value = proposal_value
            
            xs[i] = current_sample
            ys[i] = current_value

        return xs[self.burning:], ys[self.burning:]