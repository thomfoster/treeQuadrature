# Tree Quadrature
### Contact: Thomas Foster [fosterthom16@gmail.com], Yuhang Lin [daniel.kansaki@outlook.com], Otto Arend Page [otto.arendspage@jesus.ox.ac.uk]

Performs integrations of high-dimensional functions using tree-based methods.
With a set of benchmark problems defined to evaluate the performance. 
It supports various integrators, including those that use Gaussian Processes and 
distributed sampling techniques. 

The initial idea of integration using decision tree can be found in [2]. 

## Features

- Tree-based numerical integration
- Support for Gaussian Process integrators
- Distributed sampling integrators
- Parallel processing for efficient computation
- Flexible structure: user can define new trees and new integrators

## Installation
Inside the top level of this repo run 
```bash
pip install .
```

## Installation for developers
```cd``` into the top level of this repo.  
Ideally you would run all these commands inside an activated virtual environment. More on that here: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/  
To install dependencies run 
```bash
pip install -r requirements.txt
```

Install the package itself in editable mode using 
```bash
pip install -e . 
```

Don't forget the ```.```!!  
You should now be able to run a python shell inside your environment and ```import treeQuadrature```  

## Examples
A simple example: 
```python 
from treeQuadrature.example_problems import Camel
from treeQuadrature.integrators import TreeIntegrator

N = 8000 # number of samples for tree construction

problem = Camel(D=2)
integ = TreeIntegrator(N)

# perform integration
results = integ(problem, return_containers=True)

# calculate relative error 
print("error =", str(100 * (results['estimate'] - problem.answer) / problem.answer), "%")
```

See `examples/example_multivariate_normal.ipynb` on demonstration of 
- Defining a new Problem 
- Plotting the integrand of the defined Problem
- Defining and calling a tree integrator to solve this problem 
- Plot the containers and contributions obtained from the tree integrator.


## Integrators
**Tree Integrators** `TreeIntegrator` is the main integrator being used, which builds a tree and integrate containers individually. Its subclass `DistributedTreeIntegrator` that allocate samples according to volume of container and keep the number of function evaluations below a prespecified cap `max_n_samples`. Subclass of `DistributedTreeIntegrator`, `DistributedGpTreeIntegrator`, allocates samples by performance gain in GP instead. Another subclass of `TreeIntegrator`, `BatchGpIntegrator`, specialises in fitting gaussian process models to a batch of containers instead of fitting containers individually.  

**Other Integrators** Other integrators are included as baseline. Including Vegas (`VegasIntegrator`) [3], Simple Monte Carlo (`SmcIntegrator`), Bayes Monte Carlo (`BmcIntegrator`) [4], 

To define a new integrator, you can create a subclass of the `Integrator` base class and implement the necessary `__call__` method that takes a problem and other parameters, and returns a `ResultDict` with key `estiamtes`. (the estimated solution of problem) 

For more details of the integrators, please check the documentation. 

## Flexible Compartments 
This package gives you full flexibility in the tree integrators, including the following compartments. 

### Samplers

The `treeQuadrature.samplers` module provides various sampling techniques for tree-based numerical integration. These samplers are used to generate samples from the target integrand function in order to build a tree or generate samples from a container (a sub-region in the tree partition)

Some of the samplers available in `treeQuadrature` include:

- **`UniformSampler`**: This sampler generates random samples from the target distribution using a uniform distribution.

- **`SobolSampler`**: This sampler uses the Sobol sequence, a low-discrepancy sequence, to generate samples that are more evenly distributed across the input space compared to uniform sampling.

- **`LHSImportanceSampler`**: This sampler generates samples that cover the entire input space more evenly compared to random sampling, but still focus on higher density regions by the importance sampling in second stage. It uses a Latin hypercube design to ensure that each dimension of the input space is sampled uniformly and scalability to higher dimensions. 

- **`McmcSampler`**: Employs the ensemble MCMC method with multiple walks [1]. It is effective in capturing integrand functions with multiple peaks. 

For more information on each sampler and how to use them, please refer to the documentation.

### Splits

The `treeQuadrature.splits` module provides the base class `Split` for defining the splitting strategy in tree-based numerical integration.

To define a new split, you can create a subclass of the `Split` base class and implement the necessary method `split` that takes a tree container and return a list containers. This allows you to customise the behavior and properties of the split according to your specific requirements. 

Three commonly used splits are already defined in the module:

- `KdSplit`: A splitting strategy based on the k-d tree algorithm. It recursively partitions the input space by alternating between different dimensions at each level of the tree. This split is efficient for high-dimensional problems and can handle irregularly shaped regions.

- `MinSseSplit`: A splitting strategy that aims to minimise the sum of squared errors (SSE) within each partition. It iteratively evaluates different splitting points and dimensions to find the optimal split that reduces the SSE the most. This split is useful for problems where the integrand function exhibits complex patterns or non-uniform distributions.

- `UniformSplit`: The is a simple splitting strategy that divides the input space evenly into equal-sized partitions. It is suitable for problems where the integrand function is relatively smooth and evenly distributed. This split provides a relative stable tree structure, but can be time-consuming in higher dimensions

These splits can be used in conjunction with the `Tree` class (as introduced below) to build tree structures for numerical integration. 

For more information on these splits and how to use them, please refer to the documentation.

### Trees

The `treeQuadrature.trees` module provides the base class `Tree` for building the tree structures. 

To define a new tree, you can create a subclass of the `Tree` base class. Each tree should take a `split` instance and employ some stopping criteria. The functions `construct_tree` must be implemented that takes a root container and return the list of containers representing leafs of the tree. 

- `SimpleTree`: A basic tree structure. It splits containers with a queue. Each time a container is popped, and the splitted contaienrs are pushed into the queue. No priority of containers is assigned. 

- `WeightedTree`: Constructs a tree using a weighted queue mechanism for prioritising containers based on a user-defined weighting function. It supports stopping criteria through either a specified number of splits (`max_splits`) or a custom stopping condition. The class also allows for active sampling (`active_N`) before splitting containers, which can be controlled via the `split` method and the `queue`. This approach is designed to optimise the tree construction process in numerical integration tasks, with the flexibility to accommodate various splitting and prioritization strategies.

- `LimitedSampleTree`: a specialized implementation of a tree-based integrator that actively refines containers using a limited number of samples (`N`). Other aspects are similar to the `WeightedTree`. The `construct_tree` method orchestrates the refinement process, limiting the number of iterations (`max_iter`) and ensuring that the total sample count does not exceed `N`. 

For more information on these trees and how to use them, please refer to the documentations.

### Container Integrators 

The `treeQuadrature.container_integrators` module provides a variety of classes for performing numerical integration on containers (regions in the tree partition) using different methodologies. 

To define a new container integrator, you can create a subclass of the `ContainerIntegral` base class and implement the necessary `containerIntegral` method to evaluate the integral on a given container. 

Below is a list of the key classes included in this module:

- **`KernelIntegral`**: A sophisticated integrator that uses Gaussian Processes (GP) with an RBF kernel by default to estimate the integral value over a container. This method is effective for handling integrands that exhibit complex behavior, offering both an estimated integral value and an uncertainty measure. User can specify other methods of fitting Gaussian Process through defining a subclass of `treeQuadrature.gaussian_process.GPFit`, but the default is `sklearn.gaussian_process.GaussianProcessRegressor`. 

- **`AdaptiveRbfIntegral`**: An advanced integrator that adapts the length scale and search range of the RBF kernel to the specific container being integrated. This adaptive approach allows for more accurate integration in cases where the function's behavior varies significantly across the domain.

- **`RandomIntegral`**: A Monte Carlo integrator that redraws samples to estimate the integral based on a constant value. This method is useful for high-dimensional problems where traditional integration techniques may struggle. 


### Example
The following codes employs a more flexible integrator. 
```python 
import numpy as np

from treeQuadrature.example_problems import Ripple
from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.splits import MinSseSplit
from treeQuadrature.trees import WeightedTree
from treeQuadrature.container_integrators import AdaptiveRbfKernel
from treeQuadrature.samplers import McmcSampler

N = 12000 # number of samples for tree construction
max_splits = 500 # maximum number of splits allowed for tree construction

problem = Ripple(D=2)

# define a weighted tree using container volume as priority
def volume_weighting(container) -> float: 
    return container.volume
# search for best split dimension among randomly chosen dimensions (0.5 proportion)
# with probability of each dimension being chosen as the side length of container
# in that dimension
def side_length_weights(container) -> np.ndarray:
    return container.maxs - container.mins
random_split = MinSseSplit(dimension_proportion=0.5, 
                           dimension_weights=side_length_weights)
tree = WeightedTree(volume_weighting, max_splits, split=random_split)

# use Gaussian Process container integrator (with RBF kernel)
# 40 samples per container
gp_integral = AdaptiveRbfKernel(n_samples=40)

# use ensemble MCMC sampler to draw initial samples
sampler = McmcSampler()

# Combine all compartments into a TreeIntegrator
integ = TreeIntegrator(N, tree, gp_integral, sampler)

# perform integration
results = integ(problem, return_containers=True)

# calculate relative error 
print("error =", str(100 * (results['estimate'] - problem.answer) / problem.answer), "%")
```

## References

1. Foreman-Mackey, Daniel, Hogg, David W., Lang, Dustin, and Goodman, Jonathan. "emcee: The MCMC Hammer." *Publications of the Astronomical Society of the Pacific* 125, no. 925 (2013): 306-312. IOP Publishing. [doi:10.1086/670067](https://doi.org/10.1086/670067), [arXiv:1202.3665](https://arxiv.org/abs/1202.3665).

2. Foster, Thomas, Lei, Chon Lok, Robinson, Martin, Gavaghan, David, and Lambert, Ben. "Model Evidence with Fast Tree Based Quadrature." *arXiv preprint* (2020). [arXiv:2005.11300](http://arxiv.org/abs/2005.11300).

3. Lepage, G. Peter. "Adaptive Multidimensional Integration: VEGAS Enhanced." *Journal of Computational Physics* 439 (2021): 110386. [doi:10.1016/j.jcp.2021.110386](https://doi.org/10.1016/j.jcp.2021.110386), [arXiv:2009.05112](http://arxiv.org/abs/2009.05112).

4. Ghahramani, Zoubin and Rasmussen, Carl. "Bayesian Monte Carlo." *Advances in Neural Information Processing Systems* 15 (2002). MIT Press. [Link to paper](https://proceedings.neurips.cc/paper_files/paper/2002/file/24917db15c4e37e421866448c9ab23d8-Paper.pdf).