#!/usr/bin/python



def do_game(sgf, correct, tries):
    reader = SGFReader(sgf)

    if reader.komi == None:
        print "skiping %s b/c there's no komi given" % sgf
        return
    komi = float(reader.komi)
    if not komi_allowed(komi):
        print "skipping %s b/c of non-allowed komi \"%s\"" % (sgf, reader.komi)

    if reader.result == None:
        print "skipping %s because there's no result given" % sgf
        return
    elif "B+" in reader.result:
        winner = Color.Black
    elif "W+" in reader.result:
        winner = Color.White
    else:
        print "skipping %s because I can't figure out the winner from \"%s\"" % (sgf, reader.result)
        return

    turn_num = 0
    while True:
        feature_planes = feature_maker(reader.board, reader.next_play_color(), komi)
        final_score = +1 if reader.next_play_color() == winner else -1
        final_score_arr = np.array([final_score], dtype=np.int8)

        writer.push_example((feature_planes, final_score_arr))
        if reader.has_more():
            reader.play_next_move()
        else:
            break

def do_stats_on_sgfs(sgfs):
    for sgf in sgfs:
