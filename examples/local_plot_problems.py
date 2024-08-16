from treeQuadrature.visualisation import plotIntegrand
from treeQuadrature.exampleProblems import RippleProblem, OscillatoryProblem, ProductPeakProblem, CornerPeakProblem, C0Problem, DiscontinuousProblem

D = 2
problem = C0Problem(D)
plotIntegrand(problem.integrand, D=D, xlim=[problem.lows[0], problem.highs[0]], ylim = [problem.lows[1], problem.highs[1]])
# plotIntegrand(problem.integrand, D=D, xlim=[problem.lows[0], problem.highs[0]])