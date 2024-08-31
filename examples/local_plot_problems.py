from treeQuadrature.visualisation import plot_integrand
from treeQuadrature.example_problems import (
    Ripple, Oscillatory, ProductPeak, CornerPeak, C0, Discontinuous, 
    Camel, Quadratic, SimpleGaussian, ExponentialProduct, QuadCamel
)

D = 2
problem = Quadratic(D)
plot_integrand(problem.integrand, D=D, xlim=[problem.lows[0], problem.highs[0]],
               ylim = [problem.lows[1], problem.highs[1]], plot_type='contour', 
               font_size=15, tick_size=15, title=None, rotation=45)
# manual control of limits
# plot_integrand(problem.integrand, D=D, xlim=[-0.4, 0.4],
#                ylim = [-0.4, 0.4], plot_type='heat', 
#                font_size=15, tick_size=15, title=None)
# plot_integrand(problem.integrand, D=D, xlim=[problem.lows[0], problem.highs[0]])