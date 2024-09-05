from treeQuadrature.container_integrators import AdaptiveRbfIntegral
from treeQuadrature.integrators import TreeIntegrator
from treeQuadrature.example_problems import SimpleGaussian, Gaussian
from treeQuadrature.gaussian_process.visualisation import plot_gps
from treeQuadrature.visualisation import plot_containers
from matplotlib import cm


if __name__ == '__main__':
    problem = Gaussian(D=1,
                       Sigma=1/10, 
                       lows=-1,
                       highs=1)
    # problem = SimpleGaussian(D=2)

    aRbf = AdaptiveRbfIntegral(
        n_samples=3, max_redraw=0, n_splits=0, 
        fit_residuals=True)

    integ = TreeIntegrator(160, integral=aRbf)

    results, containers = integ(problem, return_raw=True,
                                return_model=True, return_std=True)

    print("Number of containers: ", len(containers))
    
    gp_models = [result['model'] for result in results]

    contributions = [result['integral'] for result in results]

    estimate = sum(contributions)

    rel_error = 100 * (estimate - problem.answer) / problem.answer
    print(f"Relative Error: {rel_error} %")

    n_evals = sum([container.N for container in containers])
    print(f"Number of evaluations: {n_evals}")

    plot_gps(gp_models, containers, title=None, alpha=0.9, 
             colormap=cm.coolwarm, plot_uncertainty=True,
             plot_samples=True)

    # if problem.D == 2:
    #     plot_containers(containers, contributions, 
    #                     xlim=[problem.lows[0], problem.highs[0]],
    #                     ylim=[problem.lows[1], problem.highs[1]], 
    #                     plot_samples=True)
    # else: 
    #     plot_containers(containers, contributions, 
    #                     xlim=[problem.lows[0], problem.highs[0]], 
    #                     integrand=problem.integrand,
    #                     plot_samples=True)
