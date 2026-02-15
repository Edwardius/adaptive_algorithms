"""
Quadratic Assignment Problem (QAP) using Tabu Search

20 departments placed in 20 locations (5 per row, 4 rows).
Minimize total cost = sum of flow(i,j) * distance(loc(i), loc(j)).
Optimal solution: 1285. 2570 double flow

Experiments:
- 20 different initial solutions (recency only, no aspiration)
- 5 different tabu list sizes
- Dynamic tabu list size
- Aspiration criteria (best-so-far override)
- Frequency-based diversification
"""

import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DISTANCE_PATH = os.path.join(SCRIPT_DIR, "q3-Distance.csv")
FLOW_PATH = os.path.join(SCRIPT_DIR, "q3-Flow.csv")


def load_matrix(path):
  """
  Read a CSV file into a 2D list of ints.

  Parameters:
    path: file path to CSV

  Returns:
    list of lists of ints
  """
  matrix = []
  with open(path, 'r') as f:
    for line in f:
      row = list(map(int, line.strip().split(',')))
      matrix.append(row)
  return matrix


NUM_SPOTS = 20
MAX_ITERATIONS = 300
OPTIMAL = 1285
DISTANCES_ARR = load_matrix(DISTANCE_PATH)
FLOWS_ARR = load_matrix(FLOW_PATH)


class QAP:
  def __init__(self, tabu_size=15, aspiration=False, dynamic_size=False,
               freq_penalty=False, freq_weight=5):
    self.tabu_table = {}
    self.freq_table = {}
    self.tabu_tenure = tabu_size
    self.tabu_size = tabu_size
    self.aspiration = aspiration
    self.dynamic_size = dynamic_size
    self.freq_penalty = freq_penalty
    self.freq_weight = freq_weight

  def find_cost(self, permutation):
    """
    Compute total flow*distance cost for a given assignment.

    Parameters:
      permutation: list where permutation[i] = department at location i

    Returns:
      total cost (int)
    """
    cost = 0
    for i in range(NUM_SPOTS):
      for j in range(i + 1, NUM_SPOTS):
        dep1, dep2 = permutation[i], permutation[j]
        cost += FLOWS_ARR[dep1][dep2] * DISTANCES_ARR[i][j]
    return cost

  def solve(self, initial_solution=None):
    """
    Run tabu search from an initial solution.

    Parameters:
      initial_solution: starting permutation, or None for random

    Returns:
      best_solution: best permutation found
      best_cost: cost of best solution
      best_iteration: iteration where best was found
    """
    if initial_solution is not None:
      current_solution = initial_solution[:]
    else:
      current_solution = random.sample(range(NUM_SPOTS), NUM_SPOTS)

    best_solution = current_solution[:]
    current_cost = self.find_cost(current_solution)
    best_cost = current_cost
    best_iteration = 0

    for iteration in range(MAX_ITERATIONS):
      best_neighbor = None
      best_neighbor_cost = float('inf')
      best_neighbor_move = (-1, -1)
      tabu_move = None

      for i in range(NUM_SPOTS):
        for j in range(i + 1, NUM_SPOTS):
          neighbor = current_solution[:]
          neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
          neighbor_cost = self.find_cost(neighbor)

          # frequency-based penalty for diversification
          if self.freq_penalty:
            neighbor_cost += self.freq_weight * self.freq_table.get((i, j), 0)

          is_tabu = (i, j) in self.tabu_table
          aspiration_override = self.aspiration and neighbor_cost < best_cost
          can_update = aspiration_override if is_tabu else neighbor_cost < best_neighbor_cost

          if can_update:
            best_neighbor = neighbor
            best_neighbor_cost = neighbor_cost
            best_neighbor_move = (i, j)
            tabu_move = (i, j) if is_tabu else None

      # dynamic tabu list size
      if self.dynamic_size:
        self.tabu_size += random.randint(-2, 2)
        self.tabu_size = max(1, self.tabu_size)

      # update tabu list tenures
      expired = []
      for key in self.tabu_table:
        self.tabu_table[key] -= 1
        if self.tabu_table[key] == 0:
          expired.append(key)
      for key in expired:
        del self.tabu_table[key]

      # add new move to tabu
      move_to_add = tabu_move if tabu_move else best_neighbor_move
      self.tabu_table[move_to_add] = self.tabu_tenure
      self.freq_table[move_to_add] = self.freq_table.get(move_to_add, 0) + 1

      # maintain tabu list size
      while len(self.tabu_table) > self.tabu_size:
        min_key = min(self.tabu_table, key=self.tabu_table.get)
        del self.tabu_table[min_key]

      if best_neighbor is not None:
        current_solution = best_neighbor
        current_cost = best_neighbor_cost

      # track best (use raw cost without freq penalty for comparison)
      raw_cost = self.find_cost(current_solution)
      if raw_cost < best_cost:
        best_solution = current_solution[:]
        best_cost = raw_cost
        best_iteration = iteration

    return best_solution, best_cost, best_iteration



def run_experiment(num_runs, **kwargs):
  """
  Run QAP solver multiple times and print results.

  Parameters:
    num_runs: number of independent runs
    **kwargs: passed to QAP constructor

  Returns:
    list of best costs from each run
  """
  print(f"\n  {'Run':<6} {'Best Cost':<12} {'Found At':<12} {'Error':<10}")
  costs = []
  for i in range(num_runs):
    qap = QAP(**kwargs)
    sol, cost, iteration = qap.solve()
    error = cost - OPTIMAL
    costs.append(cost)
    print(f"  {i+1:<6} {cost:<12} {iteration:<12} {error:<10}")
  avg = sum(costs) / len(costs)
  best = min(costs)
  print(f"  Avg cost: {avg:.1f}, Best cost: {best}, Avg error: {avg - OPTIMAL:.1f}")
  return costs


def main():
  print("Problem 3: QAP (Nugent 20) — Tabu Search")
  print(f"Locations: {NUM_SPOTS}, Iterations: {MAX_ITERATIONS}, Optimal: {OPTIMAL}")

  # Experiment 1: 20 different initial solutions, recency only, no aspiration
  print("\nEXPERIMENT 1: 20 Different Initial Solutions")
  print("Settings: tabu_size=15, no aspiration, no frequency")
  random.seed(8)
  run_experiment(20, tabu_size=15, aspiration=False,
                 dynamic_size=False, freq_penalty=False)

  # Experiment 2: Different tabu list sizes
  print("\nEXPERIMENT 2: Different Tabu List Sizes")
  print("Settings: no aspiration, no frequency, 10 runs each")
  for size in [5, 10, 15, 20, 25]:
    print(f"\n  --- Tabu list size = {size} ---")
    random.seed(88)
    run_experiment(10, tabu_size=size, aspiration=False,
                   dynamic_size=False, freq_penalty=False)

  # Experiment 3: Dynamic tabu list size
  print("\nEXPERIMENT 3: Dynamic Tabu List Size")
  print("Settings: initial size=15, varies by +/-2 each iteration, 10 runs")
  random.seed(888)
  run_experiment(10, tabu_size=15, aspiration=False,
                 dynamic_size=True, freq_penalty=False)

  # Experiment 4: Aspiration criteria
  print("\nEXPERIMENT 4: Aspiration Criteria (best-so-far override)")
  print("Settings: tabu_size=15, aspiration=True, 10 runs")
  random.seed(8888)
  run_experiment(10, tabu_size=15, aspiration=True,
                 dynamic_size=False, freq_penalty=False)

  # Experiment 5: Frequency-based diversification
  print("\nEXPERIMENT 5: Frequency-Based Diversification")
  print("Settings: tabu_size=15, aspiration=True, freq_penalty=True, weight=5, 10 runs")
  random.seed(88888)
  run_experiment(10, tabu_size=15, aspiration=True,
                 dynamic_size=False, freq_penalty=True, freq_weight=5)


if __name__ == "__main__":
  main()
