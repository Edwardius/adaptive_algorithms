"""
Problem 1: Simulated Annealing for the Easom Function

Minimize f(x) = -cos(x1)*cos(x2)*exp(-(x1-pi)^2 - (x2-pi)^2)
over x in [-100, 100]^2

Global minimum: f(pi, pi) = -1

SA Formulation:
- State: (x1, x2) in [-100, 100]^2
- Cost function: Easom function (above)
- Neighborhood: Gaussian perturbation x' = x + N(0, sigma^2), clipped to bounds
- Acceptance: Boltzmann criterion P = exp(-delta_E / T) for worsening moves
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import math

BOUNDS = (-100.0, 100.0)


def save_convergence_plot(histories, labels, title, filename):
  """
  Plot convergence curves and save to file.

  Parameters:
    histories: list of cost_history arrays, one per run
    labels: legend label for each run
    title: plot title
    filename: output file path
  """
  fig, ax = plt.subplots(figsize=(10, 6))
  for history, label in zip(histories, labels):
    ax.plot(history, label=label, alpha=0.7)
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Best Cost")
  ax.set_title(title)
  ax.legend(fontsize=7, loc="upper right")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(filename, dpi=150)
  plt.close(fig)
  print(f"  Plot saved: {filename}\n")


def easom(x1, x2):
  """
  Evaluate the Easom function at (x1, x2).

  Parameters:
    x1: first coordinate
    x2: second coordinate

  Returns:
    function value at (x1, x2)
  """
  return -math.cos(x1) * math.cos(x2) * math.exp(-((x1 - math.pi)**2 + (x2 - math.pi)**2))


def clip(val, lo=BOUNDS[0], hi=BOUNDS[1]):
  """
  Clamp val to [lo, hi].

  Parameters:
    val: value to clamp
    lo: lower bound
    hi: upper bound

  Returns:
    clamped value
  """
  return max(lo, min(hi, val))


def sa_easom(x0, T0, alpha, sigma, max_iter, schedule_type="geometric", linear_gamma=None):
  """
  Simulated Annealing for the Easom function.

  Parameters:
    x0: initial point (x1, x2)
    T0: initial temperature
    alpha: cooling rate for geometric schedule
    sigma: std dev for Gaussian neighborhood
    max_iter: number of iterations
    schedule_type: "geometric", "linear", or "logarithmic"
    linear_gamma: decrement for linear schedule

  Returns:
    best_x: best solution found
    best_cost: best cost found
    cost_history: list of best cost at each iteration
  """
  x = list(x0)
  cost = easom(x[0], x[1])
  best_x = x[:]
  best_cost = cost
  cost_history = [best_cost]

  T = T0
  if schedule_type == "linear" and linear_gamma is None:
    linear_gamma = T0 / max_iter

  for k in range(1, max_iter + 1):

    # generate neighbor
    x_new = [
      clip(x[0] + random.gauss(0, sigma)),
      clip(x[1] + random.gauss(0, sigma)),
    ]
    new_cost = easom(x_new[0], x_new[1])
    delta = new_cost - cost

    # acceptance
    if delta < 0:
      x = x_new
      cost = new_cost
    else:
      if T > 1e-15:
        p = math.exp(-delta / T)
      else:
        p = 0.0
      if random.random() < p:
        x = x_new
        cost = new_cost

    if cost < best_cost:
      best_cost = cost
      best_x = x[:]

    cost_history.append(best_cost)

    # cooling
    if schedule_type == "geometric":
      T = T * alpha
    elif schedule_type == "linear":
      T = max(T - linear_gamma, 1e-15)
    elif schedule_type == "logarithmic":
      T = T0 / (1 + math.log(1 + k))

  return best_x, best_cost, cost_history


def experiment1():
  print("\nEXPERIMENT 1: Fixed: T0=1000, alpha=0.995, sigma=30, max_iter=10000, geometric")

  T0 = 1000
  alpha = 0.995
  sigma = 30.0
  max_iter = 10000

  random.seed(42)
  np.random.seed(42)

  results = []
  histories, labels = [], []

  for i in range(10):
    x0 = (random.uniform(*BOUNDS), random.uniform(*BOUNDS))
    best_x, best_cost, history = sa_easom(x0, T0, alpha, sigma, max_iter)
    results.append((x0, best_x, best_cost))
    histories.append(history)
    labels.append(f"x0=({x0[0]:.0f},{x0[1]:.0f})")
    print(f"  Run {i+1:2d}: x0=({x0[0]:7.2f},{x0[1]:7.2f}) -> "
          f"best=({best_x[0]:.4f}, {best_x[1]:.4f}), cost={best_cost:.6f}")

  save_convergence_plot(histories, labels,
                        "Experiment 1: SA with 10 Different Initial Points",
                        "q1_experiment1.png")
  return results


def experiment2():
  print("EXPERIMENT 2: Fixed: x0=(50,50), alpha=0.995, sigma=30, max_iter=10000, geometric")

  temperatures = [0.1, 1, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
  alpha = 0.995
  sigma = 30.0
  max_iter = 10000
  x0 = (50.0, 50.0)

  results = []
  histories, labels = [], []

  for i, T0 in enumerate(temperatures):
    random.seed(42 + i)
    best_x, best_cost, history = sa_easom(x0, T0, alpha, sigma, max_iter)
    results.append((T0, best_x, best_cost))
    histories.append(history)
    labels.append(f"T0={T0}")
    print(f"  Run {i+1:2d}: T0={T0:>8.1f} -> "
          f"best=({best_x[0]:.4f}, {best_x[1]:.4f}), cost={best_cost:.6f}")

  save_convergence_plot(histories, labels,
                        "Experiment 2: SA with 10 Different Initial Temperatures",
                        "q1_experiment2.png")
  return results


def experiment3():
  print("\nEXPERIMENT 3: Fixed: x0=(50,50), T0=1000, sigma=30, max_iter=10000")

  schedules = [
    ("geometric", 0.999, None),
    ("geometric", 0.998, None),
    ("geometric", 0.995, None),
    ("geometric", 0.99, None),
    ("geometric", 0.98, None),
    ("geometric", 0.95, None),
    ("geometric", 0.90, None),
    ("linear", None, None),       # gamma auto-computed
    ("linear", None, 0.05),       # slower linear
    ("logarithmic", None, None),
  ]

  T0 = 1000
  sigma = 30.0
  max_iter = 10000
  x0 = (50.0, 50.0)

  results = []
  histories, labels = [], []

  for i, (stype, alpha_val, gamma) in enumerate(schedules):
    random.seed(42 + i)
    if stype == "geometric":
      label = f"geometric(a={alpha_val})"
      best_x, best_cost, history = sa_easom(
        x0, T0, alpha_val, sigma, max_iter, schedule_type="geometric"
      )
    elif stype == "linear":
      label = f"linear(g={gamma})" if gamma is not None else "linear(auto)"
      best_x, best_cost, history = sa_easom(
        x0, T0, 0, sigma, max_iter, schedule_type="linear", linear_gamma=gamma
      )
    elif stype == "logarithmic":
      label = "logarithmic"
      best_x, best_cost, history = sa_easom(
        x0, T0, 0, sigma, max_iter, schedule_type="logarithmic"
      )

    results.append((label, best_x, best_cost))
    histories.append(history)
    labels.append(label)
    print(f"  Run {i+1:2d}: {label:<25s} -> "
          f"best=({best_x[0]:.4f}, {best_x[1]:.4f}), cost={best_cost:.6f}")

  save_convergence_plot(histories, labels,
                        "Experiment 3: SA with 10 Different Annealing Schedules",
                        "q1_experiment3.png")
  return results


def main():
  print(f"Global minimum: f(pi, pi) = f({math.pi:.4f}, {math.pi:.4f}) = -1.0 \n")

  r1 = experiment1()
  r2 = experiment2()
  r3 = experiment3()

  best1 = min(r1, key=lambda r: r[2])
  best2 = min(r2, key=lambda r: r[2])
  best3 = min(r3, key=lambda r: r[2])

  print("\nSUMMARY: Best results from each experiment")
  print(f"  Exp 1 best: x0=({best1[0][0]:.2f}, {best1[0][1]:.2f}) -> "
        f"solution=({best1[1][0]:.6f}, {best1[1][1]:.6f}), cost={best1[2]:.6f}")
  print(f"  Exp 2 best: T0={best2[0]:.1f} -> "
        f"solution=({best2[1][0]:.6f}, {best2[1][1]:.6f}), cost={best2[2]:.6f}")
  print(f"  Exp 3 best: {best3[0]} -> "
        f"solution=({best3[1][0]:.6f}, {best3[1][1]:.6f}), cost={best3[2]:.6f}")

  # Overall best
  all_results = [(r[1], r[2]) for r in r1] + [(r[1], r[2]) for r in r2] + [(r[1], r[2]) for r in r3]
  overall_best = min(all_results, key=lambda r: r[1])
  print(f"\nOverall best: ({overall_best[0][0]:.6f}, {overall_best[0][1]:.6f}), "
        f"cost={overall_best[1]:.6f}, gap={abs(overall_best[1] - (-1.0)):.6f}")


if __name__ == "__main__":
  main()
