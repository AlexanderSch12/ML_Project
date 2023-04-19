import pyspiel
import time
from absl import app
import matplotlib.pyplot as plt

num_rows = 1
num_cols = 2
num_boxes = num_rows * num_cols
num_cells = (num_rows + 1) * (num_cols + 1)
num_parts = 3

transposition_table = {}

def part2num(part):
    p = {'h': 0, 'horizontal': 0,  # Who has set the horizontal line (top of cell)
         'v': 1, 'vertical':   1,  # Who has set the vertical line (left of cell)
         'c': 2, 'cell':       2}  # Who has won the cell
    return p.get(part, part)


def state2num(state):
    s = {'e':  0, 'empty':   0,
         'p1': 1, 'player1': 1,
         'p2': 2, 'player2': 2}
    return s.get(state, state)


def num2state(state):
    s = {0: 'empty', 1: 'player1', 2: 'player2'}
    return s.get(state, state)


def get_observation(obs_tensor, state, row, col, part):
    state = state2num(state)
    part = part2num(part)
    idx =   part \
          + (row * (num_cols + 1) + col) * num_parts  \
          + state * (num_parts * num_cells)
    return obs_tensor[idx]


def get_observation_state(obs_tensor, row, col, part, as_str=True):
    is_state = None
    for state in range(3):
        if get_observation(obs_tensor, state, row, col, part) == 1.0:
            is_state = state
    if as_str:
        is_state = num2state(is_state)
    return is_state

def _minimax(state, maximizing_player_id, alpha, beta):
    """
    Implements a min-max algorithm
    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.
    Returns:
      The optimal value of the sub-game starting in state
    """

    dbn_str = dbn_string_boxes(state)

    if dbn_str in transposition_table:
        return transposition_table[dbn_str]

    if state.is_terminal():
        return state.player_return(maximizing_player_id)

    player = state.current_player()

    if player == maximizing_player_id:
        selection = max
        # optimal_value = float('-inf')
        # for action in state.legal_actions():
        #     value = _minimax(state.child(action), maximizing_player_id, alpha, beta)
        #     optimal_value = selection(optimal_value, value)
        #     alpha = max(alpha, value)
        #     if beta <= alpha:
        #         break
    else:
        selection = min
        # optimal_value = float('inf')
        # for action in state.legal_actions():
        #     value = _minimax(state.child(action), maximizing_player_id, alpha, beta)
        #     optimal_value = selection(optimal_value, value)
        #     beta = min(beta, value)
        #     if beta <= alpha:
        #         break

    values_children = [_minimax(state.child(action), maximizing_player_id, alpha, beta) for action in state.legal_actions()]
    optimal_value = selection(values_children)

    # add state and symmetries to the transposition table
    transposition_table[dbn_str] = optimal_value
    # transposition_table[mirror_h(dbn_str)] = optimal_value
    # transposition_table[mirror_v(dbn_str)] = optimal_value
    # if num_rows  == num_cols:
    #         r1 = rotate_90_degrees(dbn_str)
    #         r2 = rotate_90_degrees(r1)
    #         r3 = rotate_90_degrees(r2)
    #         transposition_table[r1] = optimal_value
    #         transposition_table[r2] = optimal_value
    #         transposition_table[r3] = optimal_value
    # else:
    #     transposition_table[rotate_180_degrees(dbn_str)] = optimal_value

    return optimal_value


def minimax_search(game,
                   state=None,
                   maximizing_player_id=None,
                   state_to_key=lambda state: state):
    """Solves deterministic, 2-players, perfect-information 0-sum game.
    For small games only! Please use keyword arguments for optional arguments.
    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.
    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """
    game_info = game.get_type()

    if game.num_players() != 2:
        raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("The game must be a Deterministic one, not {}".format(
            game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
        raise ValueError(
            "The game must be a perfect information one, not {}".format(
                game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("The game must be turn-based, not {}".format(
            game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("The game must be 0-sum, not {}".format(game.utility))

    if state is None:
        state = game.new_initial_state()
    if maximizing_player_id is None:
        maximizing_player_id = state.current_player()
    
    v = _minimax(
        state.clone(),
        maximizing_player_id,
        float('-inf'),
        float('inf'))
    return v


def dbn_string_boxes(state):
    """Append the score for each cell to the dbn string. No box (0), player 1 box (1), player 2 box (2).
    Arguments:
        state: The current state of the game.
    Returns:
        The dbn string appended with the score for each cell
    """
    dbn_str = state.dbn_string()
    obs_tensor = state.observation_tensor(0)
    # for row in range(num_rows + 1):
    #     for col in range(num_cols):
    #         for part in ['h']:
    #             obs = get_observation_state(obs_tensor, row, col, part, False)
    #             s += str(obs)
    # for row in range(num_rows):
    #     for col in range(num_cols + 1):
    #         for part in ['v']:
    #             obs = get_observation_state(obs_tensor, row, col, part, False)
    #             s += str(obs)
    for row in range(num_rows):
        for col in range(num_cols):
            for part in ['c']:
                obs = get_observation_state(obs_tensor, row, col, part, False)
                dbn_str += str(obs)
    return dbn_str


def rotate_90_degrees(dbn_string):
    # Split the string into horizontal and vertical edges
    h_edges = dbn_string[:(num_rows + 1) * num_cols]
    v_edges = dbn_string[(num_rows + 1) * num_cols:int(len(dbn_string)) - num_boxes]
    boxes = dbn_string[int(len(dbn_string)) - num_boxes:]

    rotated_str = ""
    # v_edges to h_edges
    for i in range(int(len(v_edges) / 2)):
        rotated_str += v_edges[int(len(v_edges) / 2 - 1 - i)]
        rotated_str += v_edges[int(len(v_edges) - 1 - i)]
    # h_edges to v_edges
    for i in range(int(len(h_edges))):
        if i % 2 != 0:
            rotated_str += h_edges[i]
    for i in range(int(len(h_edges))):
        if i % 2 == 0:
            rotated_str += h_edges[i]
    # boxes
    for i in range(num_cols):
        for j in range(int(len(boxes))):
            if (j + 1) % (num_cols) == 0:
                rotated_str += boxes[j - i]
    return rotated_str


def rotate_180_degrees(dbn_string):
    # Split the string into horizontal and vertical edges
    h_edges = dbn_string[:(num_rows + 1) * num_cols]
    v_edges = dbn_string[(num_rows + 1) * num_cols:int(len(dbn_string)) - num_boxes]
    boxes = dbn_string[int(len(dbn_string)) - num_boxes:]

    rotated_str = ""
    # reverse h_edges
    rotated_str += h_edges[::-1]
    # reverse v_edges
    rotated_str += v_edges[::-1]
    # reverse boxes
    rotated_str += boxes[::-1]
    return rotated_str

def mirror_h(dbn_string):
    # Split the string into horizontal and vertical edges
    h_edges = dbn_string[:(num_rows + 1) * num_cols]
    v_edges = dbn_string[(num_rows + 1) * num_cols:int(len(dbn_string)) - num_boxes]
    boxes = dbn_string[int(len(dbn_string)) - num_boxes:]

    h_mirrored_str = ""
    # mirror h_edges
    for i in range(num_rows + 1):
        for j in range(num_cols):
            h_mirrored_str += h_edges[(i + 1) * num_cols - j - 1]
    # mirror v_edges
    for i in range(num_rows):
        for j in range(num_cols + 1):
            h_mirrored_str += v_edges[(i + 1) * (num_cols + 1) - j - 1]
    # mirror boxes
    for i in range(num_rows):
        for j in range(num_cols):
            h_mirrored_str += boxes[(i + 1) * num_cols - j - 1]
    return h_mirrored_str

def mirror_v(dbn_string):
    # Split the string into horizontal and vertical edges
    h_edges = dbn_string[:(num_rows + 1) * num_cols]
    v_edges = dbn_string[(num_rows + 1) * num_cols:int(len(dbn_string)) - num_boxes]
    boxes = dbn_string[int(len(dbn_string)) - num_boxes:]

    v_mirrored_str = ""
    # mirror h_edges
    for i in range(num_rows + 1):
        for j in range(num_cols):
            v_mirrored_str += h_edges[(num_rows - i) * num_cols + j]
    # mirror v_edges
    for i in range(num_rows):
        for j in range(num_cols + 1):
            v_mirrored_str += v_edges[(num_rows - i - 1) * (num_cols + 1) + j]
    # mirror boxes
    for i in range(num_rows):
        for j in range(num_cols):
            v_mirrored_str += boxes[(num_rows - i - 1) * num_cols + j]
    return v_mirrored_str

def plot_performance():
    naive = [0.230, 28.426]
    naive_y = [1, 2]
    transpos = [0.312, 4.565, 489.024, 33269.668]
    transpos_sym = [0.346, 6.585, 152.470, 8836.616]

    plt.plot(naive, naive_y)
    plt.show()


def main(_):
    games_list = pyspiel.registered_names()
    assert "dots_and_boxes" in games_list
    game_string = "dots_and_boxes(num_rows=" + str(num_rows) + ",num_cols=" + str(num_cols) + ")"

    print("Creating game: {}".format(game_string))
    game = pyspiel.load_game(game_string)

    start = time.time()
    value = minimax_search(game)
    end = time.time()

    print('time: ' + str((end - start) * 1000) + 'ms')
    print(str(len(transposition_table)))
    plot_performance()

    if value == 0:
        print("It's a draw")
    else:
        winning_player = 1 if value == 1 else 2
        print(f"Player {winning_player} wins.")


if __name__ == "__main__":
    app.run(main)