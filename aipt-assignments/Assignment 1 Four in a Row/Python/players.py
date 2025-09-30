from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board


class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """

        # TODO: implement minmax algortihm!
        # INT: use the functions on the 'board' object to produce a new board given a specific move
        # HINT: use the functions on the 'heuristic' object to produce evaluations for the different board states!
        
        # Example:
        # max_value: float = -np.inf # negative infinity
        # max_move: int = 0
        # for col in range(board.width):
        #     if board.is_valid(col):
        #         new_board: Board = board.get_new_board(col, self.player_id)
        #         value: int = self.heuristic.evaluate_board(self.player_id, new_board)
        #         if value > max_value:
        #             max_move = col

        # # This returns the same as
        # self.heuristic.get_best_action(self.player_id, board) # Very useful helper function!

        # This is obviously not enough (this is depth 1)
        # Your assignment is to create a data structure (tree) to store the gameboards such that you can evaluate a higher depths.
        # Then, use the minmax algorithm to search through this tree to find the best move/action to take!

        tree = board.create_game_tree(self.depth,self.player_id)
        best_move,best_value= self.minimax(tree,self.player_id,self.player_id)
        return best_move
    
    def minimax(self,game_tree: dict, player_id: int, current_player: int) -> tuple[int, int]:
        """
        Recursively performs Minimax search, alternating turns by player_id.
        Args:
            game_tree (dict): {col: (board, subtree)}
            player_id (int): the player whose best move we are computing
            heuristic (Heuristic): heuristic function
            current_player (int): player at the current node (1 or 2)
        Returns:
            tuple[int, int]: (best_move, best_value)
        """
        best_move = -1
        if current_player == player_id:
            # Maximizing player
            best_value = -np.inf
            for col, (board, subtree) in game_tree.items():
                if not subtree:  # leaf
                    value = self.heuristic.evaluate_board(player_id, board)
                else:  # recurse
                    next_player = 2 if current_player == 1 else 1
                    value = self.minimax(subtree, player_id, next_player)[1]
                if value > best_value:
                    best_value, best_move = value, col
        else:
            # Minimizing player
            best_value = np.inf
            for col, (board, subtree) in game_tree.items():
                if not subtree:  # leaf
                    value = self.heuristic.evaluate_board(player_id, board)
                else:  # recurse
                    next_player = 2 if current_player == 1 else 1
                    value = self.minimax(subtree, player_id, next_player)[1]
                if value < best_value:
                    best_value, best_move = value, col

        return best_move, best_value


class AlphaBetaPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm with alpha-beta pruning
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth

    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """

        
        tree = board.create_game_tree(self.depth,self.player_id)
        cur_depth = self.depth
        alpha = -np.inf
        beta = np.inf
        
        best_move,best_value= self.a_b_pruning(tree,self.player_id,self.player_id,alpha,beta)
        return best_move

    def a_b_pruning(self,game_tree: dict, player_id: int, current_player: int,alpha:float,beta:float) -> tuple[int, int]:

        best_move = -1

        if current_player == player_id:
            # Maximizing player
            best_value = -np.inf
            for col, (board, subtree) in game_tree.items():
                if not subtree:  # leaf
                    value = self.heuristic.evaluate_board(player_id, board)
                else:
                    next_player = 2 if current_player == 1 else 1
                    value = self.a_b_pruning(subtree, player_id, next_player, alpha, beta)[1]
                if value > best_value:
                    best_value, best_move = value, col
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # prune
        else:
            # Minimizing player
            best_value = np.inf
            for col, (board, subtree) in game_tree.items():
                if not subtree:  # leaf
                    value = self.heuristic.evaluate_board(player_id, board)
                else:
                    next_player = 2 if current_player == 1 else 1
                    value = self.a_b_pruning(subtree, player_id, next_player, alpha, beta)[1]
                if value < best_value:
                    best_value, best_move = value, col
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # prune

        return best_move, best_value
    


class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
        