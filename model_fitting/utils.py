import numpy as np
import pandas as pd
from parsers import CSVMove
from fourbynine import fourbynine_board, fourbynine_pattern, fourbynine_move, Player_Player1, Player_Player2, bool_to_player, player_to_string


def make_splits(df, n_splits  = 5, output_dir = None): 
    """
    Splits a DataFrame into a specified number of splits and optionally saves each split to a CSV file.
    Parameters:
    df (pandas.DataFrame): The DataFrame to be split.
    n_splits (int, optional): The number of splits to create. Default is 5.
    output_dir (str, optional): The directory where the CSV files will be saved. If None, the splits will not be saved to files.
    Returns:
    list: A list of DataFrames, each representing a split of the original DataFrame.
    """
    
    # shuffle the rows of the dataframe and reset the index
    df = df.sample(frac = 1).reset_index(drop = True)
    
    # Create n_splits by selecting indices separated by n_splits
    # e.g. split 1 will be taken from index 0, n_splits, 2*n_splits, etc.
    # and split 2 will be from index 1, n_splits+1, 2*n_splits+1, etc.
    splits = [df[i::n_splits].reset_index(drop = True) for i in range(n_splits)]

    # Save the splits to CSV files if output_dir is provided
    if output_dir: 
        for i, split in enumerate(splits):
            print(f"Saving split{i + 1} to {output_dir}/{i + 1}.csv")
            split.to_csv(f"{output_dir}/{i + 1}.csv", index = False)
        
    return splits

def df_to_CSVMove(df, warn = True): 
    """
    Converts a DataFrame to a generator of CSVMove objects.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to be converted. It should contain the following columns:
        - 'black': Black pieces encoded as an integer (binary encoding).
        - 'white': White pieces encoded as an integer (binary encoding).
        - 'color': The color of the player making the move ('white' or 'black').
        - 'move': The move made by the player.
        - 'reaction_time': The time taken to make the move.
        - 'group_id': The group ID of the player.
        - 'participant_id': The participant ID of the player.
        
    warn (bool, optional): If True, a warning will be printed if any of the optional columns are not present. Default is True.
    
    Yields:
    CSVMove: An object containing the following attributes:
        - board: The board configuration.
        - move: The move made by the player.
        - time: The time taken to make the move.
        - group_id: The group ID of the player.
        - participant_id: The participant ID of the player.

    Usage example
    -------------
    ```
    df = pd.read_csv('data.csv')
    for move in df_to_CSVMove(df): 
        print(move)
    ```
    """

    # Check if the required columns are present in the DataFrame
    assert 'black' in df.columns, "Column 'black' not found in DataFrame."
    assert 'white' in df.columns, "Column 'white' not found in DataFrame."
    assert 'color' in df.columns, "Column 'color' not found in DataFrame."
    assert 'move' in df.columns, "Column 'move' not found in DataFrame."

    # Print a warning if the optional columns are not present
    if warn: 
        if 'reaction_time' not in df.columns: print("Warning: Column 'reaction_time' not found in DataFrame. Defaulting to 1.0 for all moves.")
        if 'group_id' not in df.columns: print("Warning: Column 'group_id' not found in DataFrame. Defaulting to 1 for all moves.")
        if 'participant_id' not in df.columns: print("Warning: Column 'participant_id' not found in DataFrame. Defaulting to '1' for all moves.")

    for row in df.itertuples():
        board = fourbynine_board(
                    fourbynine_pattern(int(row.black)), 
                    fourbynine_pattern(int(row.white))
                    )
        
        assert row.color.lower() in ['white', 'black'], f"Invalid color given: {row.color}. Must be 'white' or 'black'."
        color = row.color.lower() == "white"

        assert color == board.active_player(), f"It's not {player_to_string(color)}'s turn to move on this board: {board.to_string()}"
        assert bin(row.move).count('1') == 1, f"Invalid move given: {row.move} does not represent a valid move (must have exactly one space occupied)."
        
        # The move is some binary number with only one bit set to 1
        # We need to find the index of the bit that is set to 1
        move_index = int(row.move).bit_length() - 1
        move = fourbynine_move(move_index, 0.0, color)

        reaction_time = row.reaction_time if 'reaction_time' in row._fields else 1.0

        group_id = row.group_id if 'group_id' in row._fields else 1
        participant_id = row.participant_id if 'participant_id' in row._fields else '1'

        yield CSVMove(board, move, reaction_time, group_id, participant_id)

