"""
Problem 4
"""

import os
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VRP_FILE = os.path.join(SCRIPT_DIR, "A-n39-k6.vrp")
OPTIMAL = 831
NUM_VEHICLES = 6


def parse_vrp(filepath):
  """
  Parse a .vrp file.

  Parameters:
    filepath: path to the .vrp file

  Returns:
    coords: dict of node_id -> (x, y)
    demands: dict of node_id -> demand
    capacity: vehicle capacity
  """
  coords = {}
  demands = {}
  capacity = 0
  section = None

  with open(filepath, 'r') as f:
    for line in f:
      line = line.strip()
      if not line or line == "EOF":
        continue
      if line.startswith("CAPACITY"):
        capacity = int(line.split(":")[-1].strip())
        continue
      if line == "NODE_COORD_SECTION":
        section = "coord"
        continue
      if line == "DEMAND_SECTION":
        section = "demand"
        continue
      if line == "DEPOT_SECTION":
        section = "depot"
        continue
      if line.startswith("NAME") or line.startswith("COMMENT") or \
         line.startswith("TYPE") or line.startswith("DIMENSION") or \
         line.startswith("EDGE_WEIGHT_TYPE"):
        continue

      parts = line.split()
      if section == "coord" and len(parts) >= 3:
        node_id = int(parts[0])
        coords[node_id] = (float(parts[1]), float(parts[2]))
      elif section == "demand" and len(parts) >= 2:
        node_id = int(parts[0])
        demands[node_id] = int(parts[1])
      elif section == "depot":
        val = int(parts[0])
        if val == -1:
          section = None

  return coords, demands, capacity


def euclidean(c1, c2):
  """
  EUC_2D distance per TSPLIB convention.

  Parameters:
    c1: (x, y) tuple
    c2: (x, y) tuple

  Returns:
    distance rounded to nearest int
  """
  return round(math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))


def save_convergence_plot(history, optimal, title, filename):
  """
  Plot convergence curve with optimal line and save to file.

  Parameters:
    history: list of best cost at each iteration
    optimal: known optimal cost (drawn as horizontal line)
    title: plot title
    filename: output file path
  """
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(history, linewidth=0.5)
  ax.axhline(y=optimal, color='r', linestyle='--', label=f'Optimal ({optimal})')
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Best Cost")
  ax.set_title(title)
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(filename, dpi=150)
  plt.close(fig)
  print(f"\n  Convergence plot saved: {filename}")


class VRP:
  def __init__(self, coords, demands, capacity, num_vehicles):
    self.coords = coords
    self.demands = demands
    self.capacity = capacity
    self.num_vehicles = num_vehicles
    self.depot = 1
    self.customers = [i for i in coords if i != self.depot]
    self.n = len(self.customers)
    # precompute distance matrix
    self.dist = {}
    all_nodes = list(coords.keys())
    for i in all_nodes:
      for j in all_nodes:
        self.dist[(i, j)] = euclidean(coords[i], coords[j])

  def perm_to_routes(self, perm):
    """
    Split a customer permutation into routes respecting capacity.

    Parameters:
      perm: list of customer ids

    Returns:
      list of routes, each a list of customer ids
    """
    routes = []
    current_route = []
    current_load = 0
    for cust in perm:
      d = self.demands[cust]
      if current_load + d > self.capacity and current_route:
        routes.append(current_route)
        current_route = [cust]
        current_load = d
      else:
        current_route.append(cust)
        current_load += d
    if current_route:
      routes.append(current_route)
    return routes

  def route_cost(self, route):
    """
    Distance for a single route: depot -> route -> depot.

    Parameters:
      route: list of customer ids

    Returns:
      total distance (int)
    """
    if not route:
      return 0
    cost = self.dist[(self.depot, route[0])]
    for i in range(len(route) - 1):
      cost += self.dist[(route[i], route[i+1])]
    cost += self.dist[(route[-1], self.depot)]
    return cost

  def total_cost(self, perm):
    """
    Total distance across all routes, with penalty for excess vehicles.

    Parameters:
      perm: customer permutation

    Returns:
      total cost (may include penalty)
    """
    routes = self.perm_to_routes(perm)
    cost = sum(self.route_cost(r) for r in routes)
    # penalize using more routes than available vehicles
    if len(routes) > self.num_vehicles:
      cost += 1000 * (len(routes) - self.num_vehicles)
    return cost

  def neighbor(self, perm):
    """
    Generate a neighbor by randomly applying swap, 2-opt, or or-opt.

    Parameters:
      perm: current customer permutation

    Returns:
      new permutation
    """
    new_perm = perm[:]
    op = random.randint(0, 2)

    if op == 0:
      # swap two random customers in the permutation
      i, j = random.sample(range(len(new_perm)), 2)
      new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    elif op == 1:
      # 2-opt: reverse a segment
      i, j = sorted(random.sample(range(len(new_perm)), 2))
      new_perm[i:j+1] = reversed(new_perm[i:j+1])
    else:
      # or-opt: move one customer to a different position
      i = random.randint(0, len(new_perm) - 1)
      cust = new_perm.pop(i)
      j = random.randint(0, len(new_perm))
      new_perm.insert(j, cust)

    return new_perm

  def solve_sa(self, T0=5000, alpha=0.9995, max_iter=200000):
    """
    Simulated Annealing for VRP.

    Parameters:
      T0: initial temperature
      alpha: geometric cooling rate
      max_iter: number of iterations

    Returns:
      best_perm: best customer permutation found
      best_cost: cost of best solution
      best_iter: iteration where best was found
      history: list of best cost at each iteration
    """
    # initial solution: random permutation of customers
    perm = self.customers[:]
    random.shuffle(perm)

    cost = self.total_cost(perm)
    best_perm = perm[:]
    best_cost = cost
    best_iter = 0
    history = [best_cost]

    T = T0
    for k in range(1, max_iter + 1):
      new_perm = self.neighbor(perm)
      new_cost = self.total_cost(new_perm)
      delta = new_cost - cost

      if delta < 0:
        perm = new_perm
        cost = new_cost
      else:
        if T > 1e-15:
          p = math.exp(-delta / T)
        else:
          p = 0.0
        if random.random() < p:
          perm = new_perm
          cost = new_cost

      if cost < best_cost:
        best_cost = cost
        best_perm = perm[:]
        best_iter = k

      history.append(best_cost)
      T *= alpha

    return best_perm, best_cost, best_iter, history


def main():
  coords, demands, capacity = parse_vrp(VRP_FILE)

  print("Problem 4: SA for Vehicle Routing Problem")
  print(f"Instance: A-n39-k6, Customers: {len(coords)-1}, Vehicles: {NUM_VEHICLES}")
  print(f"Capacity: {capacity}, Optimal: {OPTIMAL}")

  vrp = VRP(coords, demands, capacity, NUM_VEHICLES)

  # Run multiple trials
  num_trials = 5
  T0 = 5000
  alpha = 0.9995
  max_iter = 200000

  print(f"\nSA Settings: T0={T0}, alpha={alpha}, max_iter={max_iter}")
  print(f"Neighborhood: swap, 2-opt reverse, or-opt relocate")
  print(f"\n  {'Trial':<8} {'Best Cost':<12} {'Found At':<12} {'Routes':<8} {'Gap %':<10}")

  all_costs = []
  best_overall = None
  best_overall_cost = float('inf')
  best_history = None

  for trial in range(num_trials):
    random.seed(trial * 7)
    perm, cost, found_at, history = vrp.solve_sa(T0, alpha, max_iter)
    routes = vrp.perm_to_routes(perm)
    gap = (cost - OPTIMAL) / OPTIMAL * 100
    all_costs.append(cost)
    print(f"  {trial+1:<8} {cost:<12.1f} {found_at:<12} {len(routes):<8} {gap:<10.2f}")

    if cost < best_overall_cost:
      best_overall_cost = cost
      best_overall = perm
      best_history = history

  # Print best solution details
  print(f"\n  Avg cost: {sum(all_costs)/len(all_costs):.1f}")
  print(f"  Best cost: {best_overall_cost:.1f} (optimal: {OPTIMAL})")

  routes = vrp.perm_to_routes(best_overall)
  print(f"\n  Best solution routes ({len(routes)} routes):")
  for i, route in enumerate(routes):
    load = sum(vrp.demands[c] for c in route)
    dist = vrp.route_cost(route)
    route_str = " -> ".join(str(c) for c in route)
    print(f"    Route {i+1}: depot -> {route_str} -> depot "
          f"(load={load}/{capacity}, dist={dist:.1f})")

  if best_history:
    save_convergence_plot(best_history, OPTIMAL,
                          "SA Convergence for VRP (A-n39-k6)",
                          "q4_convergence.png")


if __name__ == "__main__":
  main()
