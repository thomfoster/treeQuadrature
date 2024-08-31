from treeQuadrature.visualisation import plot_integrand
from treeQuadrature.example_problems import Ripple, Oscillatory, ProductPeak, CornerPeak, C0, DiscontinuousProblem

D = 2
problem = C0(D)
plot_integrand(problem.integrand, D=D, xlim=[problem.lows[0], problem.highs[0]], ylim = [problem.lows[1], problem.highs[1]])
# plot_integrand(problem.integrand, D=D, xlim=[problem.lows[0], problem.highs[0]])