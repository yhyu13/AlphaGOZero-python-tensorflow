

# Self play games as used by DeepMind to train AlphaGo's value network. Play a
# policy against itself, but insert single random move somewhere in the game.
# Use the position immediately after the random move together with the final
# game result as a single training example for the value network.

def run_self_play_game_with_random_move(engine, N=19, komi=7.5):
    board = Board(N)

    engine.clear_board()
    engine.set_board_size(N)
    engine.set_komi(komi)

    random_policy = RandomPolicy()

    example_feature_planes = None
    example_color_to_play = None

    random_move_num = np.randint(0, 450)
    print "random_move_num = ", random_move_num
    move_num = 0
    consecutive_passes = 0
    result = None
    while consecutive_passes < 2:
        if move_num == random_move_num:
            move = random_policy.pick_move(board)
            board.play_move(move)
            engine.move_was_played(move)
            example_color_to_play = board.color_to_play
            print "chose random move (%d,%d) for %s on move #%d" % (move.x, move.y, color_names[example_color_to_play], move_num)
            example_feature_planes = Features.make_feature_planes_stones_3liberties_4history_ko(board, example_color_to_play)
        else:
            move = engine.generate_move(board)
            if move.is_resign():
                result = "B+Resign" if board.color_to_play == Color.Black else "W+Resign"
                break
            elif move.is_pass():
                consecutive_passes += 1
            else:
                consecutive_passes = 0
            board.play_move(move)
        move_num += 1

    if result == None:
        result = engine.final_score()

    print "self play game finished. result is", result

    if example_feature_planes != None:
        winner = Color.Black if "B+" in result else Color.White
        example_outcome = +1 if winner == example_color_to_play else -1
        print "produced example with example_outcome = %d" % example_outcome
        return (example_feature_planes, example_outcome)
    else:
        print "game didn't go long enough: no example produced."
        return None









