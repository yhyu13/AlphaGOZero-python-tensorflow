

class QueueWriter:
    def __init__(self, batch_queue, names, shapes, dtypes, minibatch_size, buffer_len):
        assert buffer_len >= Nperfile
        assert len(names) == len(shapes) == len(dtypes)
        self.batch_queue = batch_queue
        self.names = names
        self.shapes = shapes
        self.dtypes = dtypes
        self.minibatch_size = minibatch_size
        self.buffer_len = buffer_len
        self.examples = []

    def push_example(self, example):
        assert len(example) == len(self.names)
        for i in xrange(len(example)):
            assert example[i].dtype == self.dtypes[i]
        self.examples.append(example)
        if len(self.examples) >= self.buffer_len:
            self.write_minibatch_to_queue()

    def write_minibatch_to_queue(self):
        assert len(self.examples) >= self.minibatch_size

        # put minibatch_size random examples at the end of the list
        for i in xrange(self.minibatch_size):
            a = len(self.examples) - i - 1
            if a > 0:
              b = random.randint(0, a-1)
              self.examples[a], self.examples[b] = self.examples[b], self.examples[a]

        # pop minibatch_size examples off the end of the list
        # put each component into a separate numpy batch array
        save_dict = {}
        for c,name in enumerate(self.names):
            batch_shape = (self.minibatch_size,) + self.shapes[c]
            batch = np.empty(batch_shape, dtype=self.dtypes[c])
            for i in xrange(self.Nperfile):
                batch[i,:] = self.examples[-1-i][c]
            save_dict[name] = batch

        del self.examples[-self.Nperfile:]

        self.batch_queue.put(batch, block=True)



def make_game_data_eval(sgf, writer, feature_maker, apply_normalization, rank_allowed, komi_allowed):
    reader = SGFReader(sgf)

    if not rank_allowed(reader.black_rank) or not rank_allowed(reader.white_rank):
        print "skipping %s b/c of disallowed rank. ranks are %s, %s" % (sgf, reader.black_rank, reader.white_rank)
        return

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

    while True:
        feature_planes = feature_maker(reader.board, reader.next_play_color(), komi)
        final_score = +1 if reader.next_play_color() == winner else -1
        final_score_arr = np.array([final_score], dtype=np.int8)

        feature_planes_normalized = feature_plane.astype(np.float32)
        apply_normalization(feature_planes_normalized)

        assert False, "need to add random symmetries and maybe other stuff"

        writer.push_example((feature_planes_normalized, final_score_arr))
        if reader.has_more():
            reader.play_next_move()
        else:
            break


def async_worker_eval(self, batch_queue, sgfs, make_game_data):
    writer = QueueWriter(batch_queue=batch_queue,
            names=['feature_planes', 'final_scores'],
            shapes=[(N,N,Nfeat), (1,)],
            dtypes=[np.int8, np.int8],
            minibatch_size=128, buffer_len=50000)
    while True:
        random.shuffle(sgfs)
        for sgf in sgfs:
            make_game_data(sgf, writer)




class OnlineExampleQueue:
    def __init__(self, sgfs, make_example):
        base_dir = '/home/greg/coding/ML/go/NN/data/4dKGS/SGFs/train'
        sgfs = []
        for sub_dir in os.listdir(base_dir):
            for fn in os.listdir(os.path.join(base_dir, sub_dir)):
                    sgfs.append(os.path.join(base_dir, sub_dir, fn))

        self.q = multiprocessing.Queue(maxsize=5)

        make_game_data = functools.partial(make_game_data_eval(
            feature_maker=Features.make_feature_planes_stones_4liberties_4history_ko_4captures_komi, 
            apply_normalization=Normalization.apply_featurewise_normalization_D, 
            rank_allowed=lambda rank: rank in ['1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d'], 
            komi_allowed=lambda komi: komi in [0.5, 5.5, 6.5, 7.5])

        self.process = multiprocessing.Process(target=async_worker_eval, args=(self.q, sgfs, make_game_data))
        self.process.daemon = True
        self.process.start()

    def next_feed_dict(self, feature_planes_ph, final_scores_ph):
        feed_dict_strings = self.q.get(block=True, timeout=5)
        return { feature_planes_ph: feed_dict_strings['feature_planes'],
                 final_scores_ph: feed_dict_strings['final_scores'] }





