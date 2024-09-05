import vegas
import matplotlib.pyplot as plt
import numpy as np
from treeQuadrature.example_problems import QuadCamel
from treeQuadrature.integrators.vegas_integrator import ShapeAdapter
from treeQuadrature.integrators.vegas_tree_integrator import plot_transformed_samples

# Example integrand function
problem = QuadCamel(D=2)

integrator = vegas.Integrator([[problem.lows[0], problem.highs[0]],
                               [problem.lows[1], problem.highs[1]]])

f = ShapeAdapter(problem.integrand)
results = integrator(f, nitn=10, neval=1000)
estimate = results.mean
print(f"Relative Error {100 * (estimate-problem.answer) / problem.answer} %")

plot_transformed_samples(integrator, plot_original=True,
                         file_path='figures/vegas/transformed_samples.png',
                         n_samples=30)

# ================================
# Plot strata on the original spcae 
# ================================
grid = integrator.map.grid

# Extract the sampled points
samples = np.array([x for x, _ in integrator.random()])

# Get the adapted grid (strata)
adaptive_map = integrator.map.extract_grid()
plt.scatter(samples[:, 0], samples[:, 1], s=1, color='blue')

# Plot every nth grid line to reduce clutter
n = 5  # Plot every n'th grid line
for i in range(1, len(adaptive_map[0]) - 1, n):
    plt.axvline(x=adaptive_map[0][i], color='red', linestyle='--', linewidth=0.5)
for i in range(1, len(adaptive_map[1]) - 1, n):
    plt.axhline(y=adaptive_map[1][i], color='red', linestyle='--', linewidth=0.5)

# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.savefig('figures/vegas/stratas.png', dpi=400)
# plt.close()

# ================================
# Plot strata on transformed spcae 
# ================================
transformed_samples = np.array([y for _, y, _ in integrator.random(yield_y=True)])

plt.scatter(transformed_samples[:, 0], transformed_samples[:, 1], s=1, color='blue')

nstrat = integrator.nstrat  # This gives the number of strata in each dimension

# Plot the samples in the transformed space
transformed_samples = np.array([y for _, y, _ in integrator.random(yield_y=True)])
plt.scatter(transformed_samples[:, 0], transformed_samples[:, 1], s=1, color='blue')

# Plot the grid lines for each stratum
for i in range(1, nstrat[0]):
    plt.axvline(x=i / nstrat[0], color='red', linestyle='--', linewidth=0.5)
for i in range(1, nstrat[1]):
    plt.axhline(y=i / nstrat[1], color='red', linestyle='--', linewidth=0.5)

# Set limits for the transformed space
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.savefig('figures/vegas/stratas_transformed.png', dpi=400)
# plt.close()