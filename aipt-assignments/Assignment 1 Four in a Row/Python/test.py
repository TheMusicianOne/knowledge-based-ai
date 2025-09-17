def test_minimax_winning_move():
    from players import MinMaxPlayer
    from heuristics import SimpleHeuristic
    from board import Board
    import numpy as np
    import heuristics

    # Board layout:
    # 0-indexed columns: 0 1 2
    # X = 1, O = 2
    # Player 1 can win by placing in column 2
    state = np.array([
        [0, 0, 0],  # column 0
        [0, 0, 0],  # column 1
        [0, 0, 0]   # column 2
    ])

    board = Board(state)

    print("Board setup (player 1 can win with one move):")
    print(board)
    heuristic = SimpleHeuristic(game_n=10)
    # Initialize heuristic and MinMax player
    player_id =2
    player = MinMaxPlayer(player_id, game_n=3, depth=5, heuristic=heuristic)
    while (heuristics.Heuristic.winning(board.board_state,3)==0):
    # Get best move
        best_move = player.make_move(board)
        print(f"Best move suggested by MinMaxPlayer: {best_move}")

        # Play the move to see the board
        board.play(best_move, player.player_id)
        print("Board after playing best move:")
        print(board)
        player.player_id =1 if player.player_id ==2 else 2
        
# Run the test
test_minimax_winning_move()
