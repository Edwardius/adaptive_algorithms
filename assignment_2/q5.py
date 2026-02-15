"""
Problem 5
"""

import copy

LINES = [
  # rows
  [(0,0),(0,1),(0,2)],
  [(1,0),(1,1),(1,2)],
  [(2,0),(2,1),(2,2)],
  # columns
  [(0,0),(1,0),(2,0)],
  [(0,1),(1,1),(2,1)],
  [(0,2),(1,2),(2,2)],
  # diagonals
  [(0,0),(1,1),(2,2)],
  [(0,2),(1,1),(2,0)],
]


def initial_state():
  """
  Return an empty 3x3 board.

  Returns:
    3x3 list of None
  """
  return [[None, None, None],
          [None, None, None],
          [None, None, None]]


def legal_moves(state):
  """
  Return list of (row, col) for empty cells.

  Parameters:
    state: 3x3 board

  Returns:
    list of (row, col) tuples
  """
  moves = []
  for r in range(3):
    for c in range(3):
      if state[r][c] is None:
        moves.append((r, c))
  return moves


def apply_move(state, move, player):
  """
  Return new state with player's mark placed.

  Parameters:
    state: 3x3 board
    move: (row, col) to place mark
    player: 'X' or 'O'

  Returns:
    new 3x3 board
  """
  new_state = copy.deepcopy(state)
  new_state[move[0]][move[1]] = player
  return new_state


def check_winner(state):
  """
  Return 'X', 'O', or None.

  Parameters:
    state: 3x3 board
  """
  for line in LINES:
    vals = [state[r][c] for r, c in line]
    if vals[0] is not None and vals[0] == vals[1] == vals[2]:
      return vals[0]
  return None


def is_terminal(state):
  """
  Game is over if someone won or board is full.

  Parameters:
    state: 3x3 board
  """
  if check_winner(state) is not None:
    return True
  return all(state[r][c] is not None for r in range(3) for c in range(3))


def utility(state):
  """
  Terminal utility from X's perspective: +1 win, -1 loss, 0 draw.

  Parameters:
    state: 3x3 board (must be terminal)
  """
  winner = check_winner(state)
  if winner == 'X':
    return 1
  elif winner == 'O':
    return -1
  return 0


def evaluate(state):
  """
  Heuristic evaluation for non-terminal states.
  E(s) = #open_lines(X) - #open_lines(O)

  Parameters:
    state: 3x3 board
  """
  open_x = 0
  open_o = 0
  for line in LINES:
    vals = [state[r][c] for r, c in line]
    if 'O' not in vals:
      open_x += 1
    if 'X' not in vals:
      open_o += 1
  return open_x - open_o


def current_player(state):
  """
  Determine whose turn it is (X moves first).

  Parameters:
    state: 3x3 board
  """
  x_count = sum(1 for r in range(3) for c in range(3) if state[r][c] == 'X')
  o_count = sum(1 for r in range(3) for c in range(3) if state[r][c] == 'O')
  return 'X' if x_count == o_count else 'O'


def print_board(state):
  """
  Display the board.

  Parameters:
    state: 3x3 board
  """
  symbols = {None: '.', 'X': 'X', 'O': 'O'}
  for r in range(3):
    print("  " + " | ".join(symbols[state[r][c]] for c in range(3)))
    if r < 2:
      print("  ---------")


# ================ PART B: Depth-Limited Minimax

class MinimaxAgent:
  def __init__(self):
    self.nodes_expanded = 0

  def minimax(self, state, depth, maximizing):
    """
    Depth-limited minimax.

    Parameters:
      state: 3x3 board
      depth: remaining search depth
      maximizing: True if MAX's turn

    Returns:
      (value, best_move)
    """
    self.nodes_expanded += 1

    if is_terminal(state):
      return utility(state), None
    if depth == 0:
      return evaluate(state), None

    moves = legal_moves(state)
    best_move = None

    if maximizing:
      best_val = float('-inf')
      for move in moves:
        child = apply_move(state, move, 'X')
        val, _ = self.minimax(child, depth - 1, False)
        if val > best_val:
          best_val = val
          best_move = move
      return best_val, best_move
    else:
      best_val = float('inf')
      for move in moves:
        child = apply_move(state, move, 'O')
        val, _ = self.minimax(child, depth - 1, True)
        if val < best_val:
          best_val = val
          best_move = move
      return best_val, best_move


# ========================= PART C: Alpha-Beta Pruning

class AlphaBetaAgent:
  def __init__(self):
    self.nodes_expanded = 0

  def alphabeta(self, state, depth, alpha, beta, maximizing):
    """
    Minimax with alpha-beta pruning.

    Parameters:
      state: 3x3 board
      depth: remaining search depth
      alpha: best score MAX can guarantee
      beta: best score MIN can guarantee
      maximizing: True if MAX's turn

    Returns:
      (value, best_move)
    """
    self.nodes_expanded += 1

    if is_terminal(state):
      return utility(state), None
    if depth == 0:
      return evaluate(state), None

    moves = legal_moves(state)
    best_move = None

    if maximizing:
      best_val = float('-inf')
      for move in moves:
        child = apply_move(state, move, 'X')
        val, _ = self.alphabeta(child, depth - 1, alpha, beta, False)
        if val > best_val:
          best_val = val
          best_move = move
        alpha = max(alpha, best_val)
        if beta <= alpha:
          break
      return best_val, best_move
    else:
      best_val = float('inf')
      for move in moves:
        child = apply_move(state, move, 'O')
        val, _ = self.alphabeta(child, depth - 1, alpha, beta, True)
        if val < best_val:
          best_val = val
          best_move = move
        beta = min(beta, best_val)
        if beta <= alpha:
          break
      return best_val, best_move


# ============================ PART D: Experimental Comparison

def experiment_comparison():
  """Compare minimax vs alpha-beta at various depths from initial state."""
  print("\nPART D: Experimental Comparison — Minimax vs Alpha-Beta")

  state = initial_state()
  print("\n  Initial board (X to move):")
  print_board(state)

  depths = [2, 4, 6, 9]
  print(f"\n  {'Depth':<8} {'Algorithm':<14} {'Move':<10} {'Value':<8} {'Nodes':<10}")

  for d in depths:
    mm = MinimaxAgent()
    mm_val, mm_move = mm.minimax(state, d, True)

    ab = AlphaBetaAgent()
    ab_val, ab_move = ab.alphabeta(state, d, float('-inf'), float('inf'), True)

    print(f"  {d:<8} {'Minimax':<14} {str(mm_move):<10} {mm_val:<8} {mm.nodes_expanded:<10}")
    print(f"  {'':<8} {'Alpha-Beta':<14} {str(ab_move):<10} {ab_val:<8} {ab.nodes_expanded:<10}")

    reduction = (1 - ab.nodes_expanded / mm.nodes_expanded) * 100 if mm.nodes_expanded > 0 else 0
    print(f"  {'':<8} {'Reduction':<14} {'':<10} {'':<8} {reduction:.1f}%")
    print()


def sample_game():
  """Play a sample game: Alpha-Beta (X) vs Alpha-Beta (O), full depth."""
  print("\nSAMPLE GAME: Alpha-Beta (X, depth=9) vs Alpha-Beta (O, depth=9)")

  state = initial_state()
  move_num = 0

  while not is_terminal(state):
    player = current_player(state)
    maximizing = (player == 'X')

    agent = AlphaBetaAgent()
    val, move = agent.alphabeta(state, 9, float('-inf'), float('inf'), maximizing)

    state = apply_move(state, move, player)
    move_num += 1

    print(f"\n  Move {move_num}: {player} plays {move}, eval={val}, nodes={agent.nodes_expanded}")
    print_board(state)

  winner = check_winner(state)
  if winner:
    print(f"\n  Result: {winner} wins!")
  else:
    print(f"\n  Result: Draw!")


def main():
  print("Problem 5: Tic-Tac-Toe — Minimax and Alpha-Beta Game Search")

  # Part A: demonstrate game functions
  print("\nPART A: Game Formulation Demo")
  s = initial_state()
  print("\n  Initial state:")
  print_board(s)
  print(f"  Legal moves: {legal_moves(s)}")
  print(f"  Is terminal: {is_terminal(s)}")

  s1 = apply_move(s, (1, 1), 'X')
  print(f"\n  After X plays (1,1):")
  print_board(s1)
  print(f"  Legal moves: {legal_moves(s1)}")

  # Part B & C: show one minimax call
  print("\nPART B & C: Single-Call Demo (depth=4, from initial state)")

  mm = MinimaxAgent()
  mm_val, mm_move = mm.minimax(initial_state(), 4, True)
  print(f"\n  Minimax:    move={mm_move}, value={mm_val}, nodes={mm.nodes_expanded}")

  ab = AlphaBetaAgent()
  ab_val, ab_move = ab.alphabeta(initial_state(), 4, float('-inf'), float('inf'), True)
  print(f"  Alpha-Beta: move={ab_move}, value={ab_val}, nodes={ab.nodes_expanded}")
  print(f"  Node reduction: {(1 - ab.nodes_expanded/mm.nodes_expanded)*100:.1f}%")

  # Part D: full comparison
  experiment_comparison()

  # Sample game
  sample_game()


if __name__ == "__main__":
  main()
