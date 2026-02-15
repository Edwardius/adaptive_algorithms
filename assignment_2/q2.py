from typing import List, Tuple
from pprint import pprint
import copy
import random

"""
Implement conga on a 4x4 board

We move up/down/left/right/diagonally

## Evaluation functions

- maximize: moves(player)
- minimize: moves(opponent)
- maximize: moves(player) - moves(opponent)
"""

N = 4
P1 = "Black"
P2 = "White"
MAX_DEPTH = 3 # how many moves to see ahead
MOVE_LIMIT = 200 # how many moves before we call a "tie"
DIRECTIONS = [
  (1, 1), (1, -1), (1, 0),
  (-1, 1), (-1, -1), (-1, 0),
  (0, 1), (0, -1),
]


class Cell:
  def __init__(self, stones=0, player=None):
    self.stones = stones
    self.player = player

  def __repr__(self) -> str:
    return f"({self.player}, {self.stones})"


class CellInfo:
  def __init__(self, cell: Cell, row, col):
    self.cell = cell
    self.row = row
    self.col = col


class Move:
  """
  - d_row = number of rows we "move" up/down
  - d_col = number of cols we "move" left/right
  - row, col = (row, col) we "leave" by making the move

  move: keep "advancing" until we hit an occupied square or a wall
  """
  def __init__(self, player, row, col, direction):
    self.player = player
    self.row = row
    self.col = col
    self.direction = direction

  def __repr__(self) -> str:
    return f"(p: {self.player}, ({self.row}, {self.col}), d: {self.direction})"


def initial_grid():
  """
  Create starting 4x4 board with P1 at (0,0) and P2 at (3,3).

  Returns:
    board: NxN grid of Cells
  """
  board = [[Cell() for _ in range(N)] for _ in range(N)]
  board[0][0] = Cell(10, P1)
  board[N-1][N-1] = Cell(10, P2)
  return board


def print_grid(grid):
  """
  Print the board state.

  Parameters:
    grid: NxN grid of Cells
  """
  print("GRID:")
  for row in grid: print(row)


def get_cells(grid, row, col, direction, player) -> List[CellInfo]:
  """
  Get cells from moving in a certain direction, including
  the "start" cell.

  Parameters:
    grid: NxN grid of Cells
    row: starting row
    col: starting col
    direction: (d_row, d_col) tuple
    player: who is moving

  Returns:
    list of CellInfo along the path
  """
  d_row, d_col = direction
  result = []
  while 0 <= row < N and 0 <= col < N:
    cell_player = grid[row][col].player
    if cell_player is not None and cell_player != player:
      break
    cell_info = CellInfo(grid[row][col], row, col)
    result.append(cell_info)
    row += d_row
    col += d_col
  return result


def get_move_cells(cells) -> Tuple[CellInfo, List[CellInfo]]:
  """
  Split cell list into start and destinations.

  Parameters:
    cells: list of CellInfo from get_cells

  Returns:
    (start_cell, destination_cells)
  """
  return cells[0], cells[1:]


def game_over(grid, player):
  """
  Check if player has no legal moves.

  Parameters:
    grid: NxN grid of Cells
    player: player to check
  """
  return len(find_moves(grid, player)) == 0

def other(player):
  """
  Return the opponent.

  Parameters:
    player: current player
  """
  return P1 if player == P2 else P2


def find_moves(grid: List[List[Cell]], player) -> List[Move]:
  """
  Find list of possible moves given a grid and a player.

  Parameters:
    grid: NxN grid of Cells
    player: who is moving

  Returns:
    list of legal Move objects
  """
  result = []
  for i, row in enumerate(grid):
    for j, cell in enumerate(row):
      if cell.player != player:
        continue
      for direction in DIRECTIONS:
        d_row, d_col = direction
        if not (0 <= i+d_row < N and 0 <= j+d_col < N):
          continue
        if grid[i+d_row][j+d_col].player == other(player):
          continue
        move = Move(player, i, j, direction)
        result.append(move)
  return result


def make_move(grid: List[List[Cell]], move: Move):
  """
  Apply a move on the grid in place. Distributes 1, then 2,
  then the rest of the stones along the path.

  Parameters:
    grid: NxN grid of Cells (modified in place)
    move: the Move to apply
  """
  row, col = move.row, move.col
  direction = move.direction
  player = move.player
  stones = grid[row][col].stones
  grid[row][col] = Cell() # leave the existing cell

  _, cells = get_move_cells(get_cells(grid, row, col, direction, player))
  for i, cell_info in enumerate(cells):
    if stones == 0:
      break
    row, col = cell_info.row, cell_info.col
    stones_to_use = stones if i == len(cells) - 1 else i+1
    stones_to_use = min(stones_to_use, stones)
    grid_stones = stones_to_use
    if grid[row][col].player == player:
      grid_stones += grid[row][col].stones
    grid[row][col] = Cell(grid_stones, player)
    stones -= stones_to_use


class MinMax:
  def __init__(self, evaluate):
    self.evaluate = evaluate
    self.nodes_visited = 0

  def find_best_move(self, grid, player, depth, alpha, beta) -> Tuple[float, Move]:
    """
    Min-max with alpha-beta pruning. Higher eval = better for P1.

    Parameters:
      grid: NxN grid of Cells
      player: whose turn it is
      depth: remaining search depth
      alpha: best score MAX can guarantee
      beta: best score MIN can guarantee

    Returns:
      (score, best_move)
    """
    if depth == 0:
      self.nodes_visited += 1
      return self.evaluate(grid), None

    maximizing = (player == P1)
    result = float('-inf') if maximizing else float('inf')
    best_move = None
    if maximizing:
      moves = find_moves(grid, player)
      for candidate in moves:
        grid_copy = copy.deepcopy(grid)
        make_move(grid_copy, candidate)
        score, _ = self.find_best_move(grid_copy, other(player), depth-1, alpha, beta)
        if score > result:
          result = score
          best_move = candidate
        alpha = max(alpha, result)
        if beta <= alpha:
          break
    else:
      moves = find_moves(grid, player)
      for candidate in moves:
        grid_copy = copy.deepcopy(grid)
        make_move(grid_copy, candidate)
        score, _ = self.find_best_move(grid_copy, other(player), depth-1, alpha, beta)
        if score < result:
          result = score
          best_move = candidate
        beta = min(beta, result)
        if beta <= alpha:
          break
    return result, best_move


class RandomAgent:
  def __init__(self):
    pass

  def make_random_move(self, grid, player):
    """
    Pick a random legal move.

    Parameters:
      grid: NxN grid of Cells
      player: who is moving

    Returns:
      a random Move
    """
    moves = find_moves(grid, player)
    idx = random.randint(0, len(moves) - 1)
    return moves[idx]


def evaluate1(grid):
  """
  eval = -1 * moves(P2)

  Parameters:
    grid: NxN grid of Cells
  """
  num_moves = len(find_moves(grid, P2))
  return -num_moves

def evaluate2(grid):
  """
  eval = moves(P1)

  Parameters:
    grid: NxN grid of Cells
  """
  num_moves = len(find_moves(grid, P1))
  return num_moves

def evaluate3(grid):
  """
  eval = moves(P1) - moves(P2)

  Parameters:
    grid: NxN grid of Cells
  """
  num_moves_p1 = len(find_moves(grid, P1))
  num_moves_p2 = len(find_moves(grid, P2))
  return num_moves_p1 - num_moves_p2

def simulate_game(evaluate, print_intermediate=False):
  """
  Play a full game of MinMax (P1) vs Random (P2).

  Parameters:
    evaluate: evaluation function for MinMax agent
    print_intermediate: whether to print each turn

  Returns:
    (winner, move_count, total_nodes)
  """
  grid = initial_grid()
  move_number = 0
  total_nodes = 0

  while True:
    if print_intermediate:
      print(f"\n--- Turn {move_number} ---")
      print_grid(grid)
    player = P1 if move_number % 2 == 0 else P2
    if move_number > MOVE_LIMIT:
      if print_intermediate:
        print("move count exceeded move limit! tie game.")
      return None, move_number, total_nodes
    if game_over(grid, player):
      if print_intermediate:
        print(f"game finished! Winner: {other(player)}")
        print(f"moves made: {move_number}")
        print(f"total nodes explored by MinMax agent: {total_nodes}")
      return other(player), move_number, total_nodes

    if player == P1:
      agent = MinMax(evaluate)
      score, best_move = agent.find_best_move(grid, player, MAX_DEPTH, -float('inf'), float('inf'))
      make_move(grid, best_move)
      total_nodes += agent.nodes_visited
      if print_intermediate:
        print(f"  {player} (MinMax): move={best_move}, "
              f"eval={score}, depth={MAX_DEPTH}, nodes={agent.nodes_visited}")
    elif player == P2:
      agent = RandomAgent()
      random_move = agent.make_random_move(grid, player)
      make_move(grid, random_move)
      if print_intermediate:
        print(f"  {player} (Random): move={random_move}")
    move_number += 1



def main():
  num_trials = 5
  eval_fns = [
    (evaluate1, "evaluate1: -moves(opponent)"),
    (evaluate2, "evaluate2: moves(player)"),
    (evaluate3, "evaluate3: moves(player) - moves(opponent)"),
  ]

  print("Problem 2")
  print(f"Board: {N}x{N}, Max Depth: {MAX_DEPTH}, Move Limit: {MOVE_LIMIT}")

  # Run one detailed game with evaluate3 for sample output
  print("\nSAMPLE GAME")
  random.seed(0)
  winner, moves, nodes = simulate_game(evaluate3, print_intermediate=True)

  # Run experiments for all eval functions
  print(f"\nRESULTSS ({num_trials} trials per evaluation function)")

  for eval_fn, eval_name in eval_fns:
    print(f"\n  {eval_name}")
    print(f"  {'Trial':<8} {'Winner':<10} {'Moves':<10} {'Nodes':<15}")
    wins = {P1: 0, P2: 0, None: 0}
    total_moves = []
    total_nodes_list = []
    for trial in range(num_trials):
      random.seed(trial)
      winner, moves, nodes = simulate_game(eval_fn)
      wins[winner] += 1
      total_moves.append(moves)
      total_nodes_list.append(nodes)
      winner_str = winner if winner else "Tie"
      print(f"  {trial+1:<8} {winner_str:<10} {moves:<10} {nodes:<15}")

    print(f"  Summary: Black wins={wins[P1]}, White wins={wins[P2]}, Ties={wins[None]}")
    print(f"  Avg moves: {sum(total_moves)/len(total_moves):.1f}, "
          f"Avg nodes: {sum(total_nodes_list)/len(total_nodes_list):.0f}")


if __name__ == "__main__":
  main()
