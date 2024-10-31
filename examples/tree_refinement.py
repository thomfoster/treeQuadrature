from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.trees import SimpleTree
from treeQuadrature.example_problems import Problem, SimpleGaussian
from treeQuadrature import Container
from treeQuadrature.visualisation import plot_containers

from typing import List, Optional
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def refine_tree(
    integrator: TreeIntegrator, 
    problem: Problem, 
    num_splits: List[int],
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
    num_splits : list[int]
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
    # Initial root container or use provided containers
    if containers is None:
        root = integrator._construct_root_container(
            *integrator._draw_initial_samples(
                problem, verbose=False),
            problem, verbose=False
        )
        leaf_containers = [root]
    else:
        leaf_containers = containers
    
    errors = {}
    total_splits = len(leaf_containers)
    max_num_split = max(num_splits)

    # Incrementally refine the tree and test accuracy
    for i, splits in enumerate(num_splits):
        leaf_containers = integrator.tree.construct_tree(
            root=leaf_containers, max_iter=max(max_num_split, total_splits),
            verbose=False,
            max_splits=splits,
            warning=False
        )
        if len(leaf_containers) - total_splits == 0:
            print("No new containers splitted, stopping the refinement")
            break
        total_splits = len(leaf_containers)
        
        # Perform integration on the refined tree
        results = integrator.integrate_containers(leaf_containers, problem)
        contributions = [result["integral"] for result in results[0]]
        estimate = sum(contributions)
        relative_error = (estimate - problem.answer) / problem.answer
        errors[total_splits] = relative_error
        if verbose:
            print(
                f"Splits: {total_splits}, "
                f"Estimate: {estimate}, "
                f"Relative error: {relative_error:.4%}")
            
        # TODO - test codes to be deleted
        plot_containers(leaf_containers, contributions,
                        xlim=[-0.4, 0.4],
                        ylim=[-0.4, 0.4],
                        file_path=f"figures/tree_splitting/containers_{i}.png",
                        title=f'{total_splits} splits')
        
        plt.close()
    
    return errors


if __name__ == '__main__':
    D = 2
    N = int(20_000 * (D/3))
    integrator = TreeIntegrator(N, tree=SimpleTree(P=20))
    problem = SimpleGaussian(D)
    num_splits = [5] * 20

    filenames = [f"figures/tree_splitting/containers_{i}.png"
                 for i in range(len(num_splits))]

    errors = refine_tree(integrator, problem, num_splits)

    # Create a GIF from the saved images
    with imageio.get_writer('figures/tree_construction.gif',
                            mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
