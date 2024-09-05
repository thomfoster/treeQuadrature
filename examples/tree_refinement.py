from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.trees import SimpleTree
from treeQuadrature.example_problems import Problem, SimpleGaussian
from treeQuadrature import Container 

from typing import List, Optional


def refine_tree(
    integrator: TreeIntegrator, 
    problem: Problem, 
    additional_splits: List[int],
    containers: Optional[List[Container]]=None, 
    verbose:bool=True
):
    """
    Refine an existing tree or containers and
    test the accuracy of the integration
    as more splits are added.

    Parameters
    ----------
    integrator : TreeIntegrator
        An instance of TreeIntegrator or a subclass.
    problem : Problem
        The integration problem to be solved.
    additional_splits : list[int]
        A list of additional splits to be performed
        after the initial tree construction.
    containers : list[Container], optional
        A list of existing containers to continue refining. \n
        If None, the tree is constructed from the root.
    verbose : bool, optional
        If True, print progress and results
        during the refinement process.

    Returns
    -------
    dict
        A dictionary where keys are cumulative
        splits and values are the
        relative errors of the integration results.
    """
    # Initial tree construction or use provided containers
    if containers is None:
        root = integrator._construct_root_container(
            *integrator._draw_initial_samples(
                problem, verbose=False),
            problem, verbose=False
        )
        leaf_containers = integrator._construct_tree(
            root, problem, verbose=verbose)
    else:
        leaf_containers = containers
    
    errors = {}
    total_splits = 0

    # Perform initial integration
    results = integrator.integrate_containers(
        leaf_containers, problem)
    estimate = sum(result["integral"] for result in results[0])
    relative_error = (estimate - problem.answer) / problem.answer
    errors[total_splits] = relative_error
    if verbose:
        print(
            f"Splits: {total_splits}, "
            f"Estimate: {estimate}, "
            f"Relative error: {relative_error:.4%}")

    # Incrementally refine the tree and test accuracy
    for splits in additional_splits:
        total_splits += splits

        # add more samples
        for cont in leaf_containers:
            xs = cont.rvs(10)
            ys = problem.integrand(xs)
            cont.add(xs, ys)
        
        leaf_containers = integrator.tree.construct_tree(
            root=leaf_containers, max_iter=splits,
            verbose=verbose, warning=False
        )
        
        # Perform integration on the refined tree
        results = integrator.integrate_containers(leaf_containers, problem)
        estimate = sum(result["integral"] for result in results[0])
        relative_error = (estimate - problem.answer) / problem.answer
        errors[total_splits] = relative_error
        if verbose:
            print(
                f"Splits: {total_splits}, "
                f"Estimate: {estimate}, "
                f"Relative error: {relative_error:.4%}")
    
    return errors


if __name__ == '__main__':
    tree = SimpleTree()
    integrator = TreeIntegrator(10_000, tree=tree)

    problem = SimpleGaussian(D=2)
    additional_splits = [100] * 5

    refine_tree(integrator, problem, additional_splits)