from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from .sampler import Sampler
from ..exampleProblems import Problem

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
    the modulus of problem.integrand
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

    def rvs(self, n: int, problem: Problem) -> np.ndarray:
        """
        Generate MCMC samples.

        Parameters
        ----------
        n : int 
            Number of samples.
        problem: Problem
            The integration problem being solved.

        Returns
        -------
        np.ndarray
            Samples from the modulus of the integrand.
        """
        D = problem.D
        samples = np.zeros((n+self.burning, D))
        current_sample = np.random.uniform(low=problem.lows, 
                                           high=problem.highs, size=D)
        
        for i in range(n+self.burning):
            proposal = self.proposal.propose(current_sample, 
                                             problem.lows, problem.highs)
            proposal = np.clip(proposal, problem.lows, problem.highs)
            
            current_value = np.abs(problem.integrand(current_sample.reshape(1, -1)))[0]
            proposal_value = np.abs(problem.integrand(proposal.reshape(1, -1)))[0]

            proposal_density_forward = self.proposal.density(proposal, current_sample)
            proposal_density_backward = self.proposal.density(current_sample, proposal)

            acceptance_ratio = min(1, (proposal_value / current_value) * 
                                   (proposal_density_backward / proposal_density_forward))
            
            if np.random.rand() < acceptance_ratio:
                current_sample = proposal
            
            samples[i] = current_sample

        return samples[self.burning:]