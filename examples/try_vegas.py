import vegas
import matplotlib.pyplot as plt
import numpy as np
from treeQuadrature.example_problems import QuadCamel
from treeQuadrature.integrators.vegas_integrator import ShapeAdapter

# Example integrand function
problem = QuadCamel(D=2)

integrator = vegas.Integrator([[problem.lows[0], problem.highs[0]],
                               [problem.lows[1], problem.highs[1]]])

f = ShapeAdapter(problem.integrand)
results = integrator(f, nitn=10, neval=1000)
estimate = results.mean
print(f"Relative Error {100 * (estimate-problem.answer) / problem.answer} %")

# Extract the adaptive grid
grid = integrator.map.grid

# Extract the sampled points
samples = np.array([x for x, _ in integrator.random()])

# Get the adapted grid (strata)
adaptive_map = integrator.map.extract_grid()
plt.scatter(samples[:, 0], samples[:, 1], s=1, color='blue')

# Plot every nth grid line to reduce clutter
n = 10  # Plot every 10th grid line
for i in range(1, len(adaptive_map[0]) - 1, n):
    plt.axvline(x=adaptive_map[0][i], color='red', linestyle='--', linewidth=0.5)
for i in range(1, len(adaptive_map[1]) - 1, n):
    plt.axhline(y=adaptive_map[1][i], color='red', linestyle='--', linewidth=0.5)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()