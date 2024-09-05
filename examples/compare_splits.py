from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.trees import SimpleTree
from treeQuadrature.samplers import McmcSampler
from treeQuadrature.splits import KdSplit, UniformSplit
from treeQuadrature.example_problems import Camel
from treeQuadrature import Container
from treeQuadrature.visualisation import plot_containers


if __name__ == '__main__':
    N = 8_000

    mcmcSampler = McmcSampler()

    problem = Camel(D=2)

    tree_kd = SimpleTree(split=KdSplit())
    tree_uniform = SimpleTree(split=UniformSplit())

    ### Set integrators
    integ_mean_minsse = TreeIntegrator(
        N, sampler=mcmcSampler)

    integ_mean_kd = TreeIntegrator(
        N, tree=tree_kd, sampler=mcmcSampler)

    integ_mean_uniform = TreeIntegrator(
        N, tree=tree_uniform, sampler=mcmcSampler)

    integrators = [integ_mean_minsse, integ_mean_kd, integ_mean_uniform]

    split_names = ['minsse', 'kd', 'uniform']

    xs, ys = integ_mean_minsse._draw_initial_samples(problem, False)
    root = Container(xs, ys, problem.lows, problem.highs)
    plot_containers([root], None, plot_samples=True,
                    xlim=[0, 1], ylim=[0, 1],
                    file_path=f'figures/splits/samples.png')

    for i, integ in enumerate(integrators):
        results = integ(problem, return_containers=True)

        plot_containers(results['containers'], None,
                        xlim=[0, 1], ylim=[0, 1],
                        file_path=f'figures/splits/{split_names[i]}_split_camel.png',
                        resolution=500)