import pandas as pd
import numpy as np
from scipy.stats import entropy


def create_game_tree():
    stack = [("", " " * 9, 0, 1.0, [])]  # (prev_board, board, depth, prob, moves)
    result = stack[:]
    while stack:
        _, board, depth, prob, moves = stack.pop()
        if is_game_over(board):
            continue
        move_char = "X" if depth % 2 == 0 else "O"
        valid_moves = [i for i in range(9) if board[i] == " "]
        move_prob = 1.0 / len(valid_moves)
        for move in valid_moves:
            new_board = board[:move] + move_char + board[move + 1 :]
            next_node = (board, new_board, depth + 1, move_prob * prob, moves + [move])
            stack.append(next_node)
            result.append(next_node)
    return result


def is_game_over(board):
    return (
        board.count(" ") == 0
        or any(
            board[i] != " " and board[i : i + 3].count(board[i]) == 3
            for i in range(0, 9, 3)
        )
        or any(
            board[i] != " " and board[i::3].count(board[i]) == 3 for i in range(0, 3)
        )
        or board[0] != " "
        and board[0::4].count(board[0]) == 3
        or board[2] != " "
        and board[2:8:2].count(board[2]) == 3
    )


def print_board(board):
    print("\n".join(board[i : i + 3] for i in range(0, 9, 3)))


def create_game_dataframe():
    tree = create_game_tree()
    game_df = pd.DataFrame(
        tree, columns=["prev_board", "board", "depth", "prob", "moves"]
    )

    prob_tot = game_df.groupby("depth").prob.sum()
    game_df = game_df.assign(prob_norm=lambda x: x.prob / x.depth.map(prob_tot))

    state_ids = {
        board: i
        for i, board in enumerate(
            sorted(
                game_df.board.unique(), key=lambda x: (x.count(" "), x), reverse=True
            )
        )
    }

    game_df = game_df.assign(state_id=lambda x: x.board.map(state_ids)).assign(
        is_game_over=lambda x: x.board.apply(is_game_over)
    )

    return game_df


def get_full_sequences(game_df, pad_token=-1, bos=None):
    depth = 9
    df = game_df.query('depth == @depth | (depth < @depth & is_game_over)')
    sequences = np.array([np.pad(moves, (0, 9 - len(moves)), constant_values=pad_token) for moves in df.moves])
    
    if isinstance(bos, int):
        sequences = np.pad(sequences, ((0, 0), (1, 0)), constant_values=bos)
    
    probabilities = df.prob.to_numpy()
    return sequences, probabilities
def compute_block_entropy(game_df):
    block_entropy = []
    for depth in range(10):
        probs = game_df.query("depth == @depth | (depth < @depth & is_game_over)").prob
        assert np.isclose(probs.sum(), 1)
        block_entropy.append(entropy(probs))
    return block_entropy


def compute_myopic_entropy(block_entropy):
    return np.diff(block_entropy)


# Example usage
if __name__ == "__main__":
    game_df = create_game_dataframe()
    block_entropy = compute_block_entropy(game_df)
    myopic_entropy = compute_myopic_entropy(block_entropy)

    print("Block Entropy:", block_entropy)
    print("Myopic Entropy:", myopic_entropy)
