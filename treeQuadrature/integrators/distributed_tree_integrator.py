from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import Optional, List
from inspect import signature
import warnings

from .tree_integrator import TreeIntegrator
from ..container_integrators import ContainerIntegral
from ..trees import Tree
from ..samplers import Sampler
from ..example_problems import Problem
from ..container import Container


def integral_wrapper(
    integral: ContainerIntegral,
    cont: Container,
    integrand: callable,
    return_std: bool,
    n_samples: int,
    **kwargs
):
    """
    wrapper function of container_integral for parallel processing
    with essential checks on the results
    """
    try:
        _ = integral.n_samples
    except AttributeError:
        raise AttributeError(
            "self.integral does not have attribute "
            "n_samples, cannot use DistributedTreeIntegrator"
        )

    integral.n_samples = n_samples

    params = {}
    if hasattr(integral, "get_additional_params"):
        params.update(integral.get_additional_params())

    params.update(kwargs)

    if return_std:
        params["return_std"] = True

    integral_results = integral.containerIntegral(cont, integrand, **params)

    # check types
    if "integral" not in integral_results:
        raise KeyError(
            "results of containerIntegral does not have key 'integral'"
        )
    elif return_std and "std" not in integral_results:
        raise KeyError(
            "results of containerIntegral does not have key 'std'"
        )

    return integral_results, cont


class DistributedTreeIntegrator(TreeIntegrator):
    def __init__(
        self,
        base_N: int,
        max_n_samples: int,
        integral: Optional[ContainerIntegral] = None,
        sampler: Optional[Sampler] = None,
        tree: Optional[Tree] = None,
        scaling_factor: float = 1e-6,
        min_container_samples: int = 2,
        max_container_samples: int = 200,
        parallel: bool = True,
    ) -> None:
        """
        A TreeIntegrator that constructs a tree and then distributes the
        remaining samples among the containers obtained
        according to the volume of containers.

        Attributes
        ----------
        base_N : int
            Total number of initial samples.
        integral : ContainerIntegral
            Method to evaluate the integral of f on a container.\n
            Default: RandomIntegral.
        sampler : Sampler, optional
            Method for generating initial samples,
            when the problem does not have an rvs method.
        tree : Tree, optional
            Method to construct the tree of containers.
            Default: SimpleTree.
        max_n_samples : int
            Total number of evaluations available.
        min_container_samples, max_container_samples: int, optional
            The minimum and maximum number of samples to
            allocate to each container. \n
            Default: 2 and 200
        scaling_factor : float, optional
            A scaling factor to control the aggressiveness
            of sample distribution. \n
            Default: 1e-6
        parallel : bool, optional
            whether to use parallel computing for container integration
        """
        super().__init__(base_N, tree, integral, sampler, parallel)
        self.max_n_samples = max_n_samples
        self.max_container_samples = max_container_samples
        self.min_container_samples = min_container_samples
        self.scaling_factor = scaling_factor

    def integrate_containers(
        self,
        containers: List[Container],
        problem: Problem,
        compute_std: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        # Determine the number of remaining samples to distribute
        used_samples = np.sum([cont.N for cont in containers])
        remaining_samples = self.max_n_samples - used_samples

        # Maximum value of min_container_samples that can be used
        upper_cap = (
            self.max_n_samples - used_samples) // len(containers)

        if upper_cap >= self.min_container_samples:
            # No need to adjust,
            # the condition is already satisfied
            pass
        elif upper_cap >= 2:
            warnings.warn(
                f"Too many samples to distribute. \n"
                "Reducing 'min_container_samples' "
                f"from {self.min_container_samples} to {upper_cap}."
            )
            self.min_container_samples = upper_cap
        else:
            raise RuntimeError(
                "Too many samples to distribute. "
                "Even with 'min_container_samples = 2', "
                "the condition cannot be satisfied. "
                "Consider increasing 'max_n_samples', "
                "or reduce the number of containers."
            )

        if remaining_samples > 0:
            samples_distribution = self._distribute_samples(
                containers,
                remaining_samples,
                self.min_container_samples,
                self.max_container_samples,
                problem.D,
            )

        total_samples = sum(samples_distribution.values())
        if total_samples > remaining_samples:
            raise RuntimeError(
                "allocated too many samples"
                f"upper limit: {remaining_samples}"
                f"allocated samples: {total_samples}"
            )

        if verbose:
            print(
                "Integrating individual containers",
                "with standard deviation" if compute_std else "",
            )
            print(
                "largest container distribution: "
                f"{max(samples_distribution.values())}"
            )
            print(
                "smallest container distribution: "
                f"{min(samples_distribution.values())}"
            )

        # for retracking containers
        modified_containers = []
        results = []

        integrator_parameters = signature(
            self.integral.containerIntegral).parameters

        applicable_kwargs = {
            k: v for k, v in kwargs.items()
            if k in integrator_parameters
        }

        if self.parallel:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        integral_wrapper,
                        self.integral,
                        cont,
                        problem.integrand,
                        compute_std,
                        n_samples=samples_distribution.get(cont),
                        **applicable_kwargs
                    ): cont
                    for cont in containers
                }

                for future in as_completed(futures):
                    integral_results, modified_cont = future.result()
                    results.append(integral_results)
                    modified_containers.append(modified_cont)
        else:
            for cont in containers:
                integral_results, modified_cont = integral_wrapper(
                    self.integral,
                    cont,
                    problem.integrand,
                    compute_std,
                    samples_distribution.get(cont),
                    **applicable_kwargs
                )
                results.append(integral_results)
                modified_containers.append(modified_cont)

        return results, modified_containers

    @staticmethod
    def _distribute_samples(
        finished_containers,
        remaining_samples,
        min_container_samples,
        max_container_samples,
        problem_dim,
    ):
        total_volume = sum(c.volume for c in finished_containers)
        samples_distribution = {}

        # Initial distribution based on scaled volume
        total_assigned = 0
        for cont in finished_containers:
            scaled_volume = (cont.volume / total_volume) ** (1 / problem_dim)
            # Calculate initial allocation
            additional_samples = max(
                min_container_samples, int(remaining_samples * scaled_volume)
            )
            # Cap at max_container_samples and ensure minimum allocation
            additional_samples = min(max_container_samples, additional_samples)

            samples_distribution[cont] = additional_samples
            total_assigned += additional_samples

        # Adjust allocations if too many samples were assigned
        if total_assigned > remaining_samples:
            excess = total_assigned - remaining_samples
            for cont in sorted(finished_containers, key=lambda c: c.volume):
                if excess <= 0:
                    break
                reduce_by = min(
                    excess, samples_distribution[cont] - min_container_samples
                )
                samples_distribution[cont] -= reduce_by
                excess -= reduce_by

        # Re-check and distribute any leftover samples if less were assigned
        if total_assigned < remaining_samples:
            remainder_samples = remaining_samples - total_assigned
            for cont in sorted(finished_containers, key=lambda c: -c.volume):
                if remainder_samples <= 0:
                    break
                if samples_distribution[cont] < max_container_samples:
                    samples_to_add = min(
                        remainder_samples,
                        max_container_samples - samples_distribution[cont],
                    )
                    samples_distribution[cont] += samples_to_add
                    remainder_samples -= samples_to_add

        # Final check to ensure we never allocate more than allowed
        if sum(samples_distribution.values()) > remaining_samples:
            raise RuntimeError("Allocated too many samples")

        return samples_distribution
