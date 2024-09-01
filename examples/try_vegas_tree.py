from treeQuadrature.integrators.vegas_tree_integrator import VegasTreeIntegrator
from treeQuadrature.container_integrators import AdaptiveRbfIntegral, RandomIntegral
from treeQuadrature.example_problems import (
    Camel, SimpleGaussian, QuadCamel
)
from treeQuadrature.compare_integrators import compare_integrators

if __name__ == '__main__':
    problem = QuadCamel(D=2)
    
    aRBF = AdaptiveRbfIntegral(
        n_samples=30, max_redraw=0, n_splits=0)
    integ_vegas_rbf = VegasTreeIntegrator(
        tree_N=3000, vegas_N=8000, integral=aRBF)
    integ_vegas_rbf.name = 'VEGAS + Tree + RBF'

    random_integral = RandomIntegral(n_samples=30)
    integ_vegas_mean = VegasTreeIntegrator(
        tree_N=3000, vegas_N=8000, integral=random_integral)
    integ_vegas_mean.name = 'VEGAS + Tree + Mean'

    # results = integ_vegas_mean(
    #     problem, return_N=True, return_containers=True,
    #     plot_vegas=True, file_path='figures/vegas_map_quad.png')

    compare_integrators([integ_vegas_mean], problem,
                        plot=True, xlim=[0, 1], ylim=[0, 1],
                        n_repeat=1, title=None, plot_vegas=True, 
                        file_path='figures/vegas_map_quad.png')
