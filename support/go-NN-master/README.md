## Go-playing neural network in Python using TensorFlow

This project is a replication of the policy network from DeepMind's 
[AlphaGo paper](http://airesearch.com/wp-content/uploads/2016/01/deepmind-mastering-go.pdf).

I don't play go, but I got really excited about neural networks in go after watching the
[AlphaGo-Lee Sedol match](https://www.youtube.com/playlist?list=PLqYmG7hTraZA7v9Hpbps0QNmJC4L1NE3S).
To me the most interesting part is that you can train a policy network to play pretty good
go with no lookahead search at all. This is very different from chess where most of the
intelligence of chess engines comes from deep tree search.

### Papers on neural networks in go

The AlphaGo paper was preceded by a number of other papers on neural networks in go. I found
these papers useful:

* AlphaGo paper, 2016 - [Nature version](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html), [free preprint](http://airesearch.com/wp-content/uploads/2016/01/deepmind-mastering-go.pdf)
* [Tian and Zhu](http://arxiv.org/pdf/1511.06410v3.pdf) - on Facebook's Darkforest go player
* [Maddison et al.](http://arxiv.org/pdf/1412.6564v2.pdf) - an earlier paper by some AlphaGo authors
* [Clark and Storkey](http://arxiv.org/pdf/1412.3409v2.pdf)

### Training data

I bought the GoGoD database (Winter 2015 version, just $15) of over 85,000 professional go games to train
the neural network. The neural network was trained on about 140 million position-move pairs from
these games. I excluded games before 1800 AD. Some games include amateur players; I ignored
the moves of amateurs. I excluded games that include illegal moves (apparently sometimes even top
pros make illegal moves and so forfeit the game!). I also tried to exclude games where the comments
suggest that the game record is corrupt.

The input to the neural network is 21 feature planes, each a 19x19 binary image. The feature planes are

* Stones belonging to the player whose turn it is
* Stones belonging to the opponent
* Empty points
* A plane of all ones (useful for detecting board edge, since convolutions are zero-padded)
* Four planes encoding the liberty count of the groups of the player whose turn it is. If a group
has one liberty, its stones are turned on in the first plane. If a group has two liberties, its
stones are turned on in the second plane... If a group has four or more liberties, its stones are
turned on in the fourth plane
* Four planes encoding the liberty count of the opponent's groups in the same way.
* Four planes giving the locations of the last four moves (one point is turned on in each plane)
* One plane where a bit is turned on if it is illegal to play there by the ko rule.
* Four planes encoding the number of stones that would be captured by playing at a given point.
If one stone would be captured, the point is turned on in the first plane... If four or more stones
would be captured, the point is turned on in the fourth plane.

### Neural network architecture

I implemented and trained the neural network using Google's [TensorFlow](https://www.tensorflow.org/) library.

I used a very similar architecture to the AlphaGo policy network. The network is fully convolutional
with twelve layers. The first layer is a 5x5 convolution and the rest are 3x43 convolutions. Each layer
has 256 filters, except the last layer which has only one filter. So the output is a single 19x19 plane
which is fed to a softmax to obtain a probability distribution over moves.

Unlike AlphaGo, I use position-dependent biases in each layer. I think AlphaGo just uses position dependent
biases in the last layer. I didn't test the effect of this change.

I use the ELU activation function instead of ReLU. Some tests suggested that this sped up training somewhat.

The neural network literature suggests that training is faster if the inputs to the network are normalized,
but I haven't seen any mention of input normalization in the papers above. 
I normalize the inputs to the neural network by calculating the mean and standard deviation of each feature over 
the training data. I subtract out the mean of each feature and rescale by one over the standard deviation,
except I don't rescale any feature by more than a factor of ten. So all the inputs have mean 0, but some
have standard deviations less than one. I observed a modest speedup in training from doing this.

### Training

After doing a bunch of experiments. I trained the final network for a week on a GTX 980 Ti using stochastic gradient descent
with momentum. None of the papers above use momentum, but I'm not sure why. I observed a small benefit from
using a momentum of 0.9 over using no momentum.

The loss function was the average negative log-likelihood of the correct move (the one played by
the pro in each position).

TODO: add some charts of learning progress, give final accuracy values.

### Playing on KGS

I implemented the [Go Text Protocol](https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html) so
that I could have the network play on the [KGS go server](https://www.gokgs.com/). 

One problem was that the neural network is only trained to generate moves; it doesn't know how to pass at the
end of the game, or how to score a position. So I duct taped GnuGo to the neural network to handle those parts.
When playing a game, we first ask GnuGo to generate a move. If GnuGo passes, we pass. Otherwise we ignore GnuGo's
move and instead play the top move suggested by the network. Commands related to scoring are passed through to GnuGo.

The network plays on KGS under the name QuarkBot. I am surprised how popular it has been: people play it almost nonstop 24/7.
Based on its performance in unranked games, I estimate that it plays at around the KGS 1 dan level. It is probably not quite as strong
as AlphaGo's policy network, but it is pretty close.

### Thoughts

* I am extremely impressed that I, someone who barely knows how to play go, can download a bunch of data, train a
generic machine learning algorithm for a week, and obtain a strong go player. A KGS rating of 1 dan is around the
[85th percentile](http://senseis.xmp.net/?RatingHistogramComparisons) of 
[ratings on the KGS server](http://senseis.xmp.net/?KGSRankHistogram). I'm also impressed that such strong play 
can be achieved with literally no lookahead search at all.
* I have now spent many hours watching the neural network play people on KGS, and I like to think I've gotten
a bit better at go by doing so. Often I feel like I can predict the players' moves! 
After all, that's how the network learned: watching the games of strong players and trying to predict their moves.
* I was limited in this project by computing power. AlphaGo's policy network was trained on 50 GPUs for three weeks.
It was improved by reinforcement learning through self play on 50 GPUs for one day. Its value network was
trained on 50 GPUs for one week. Overall that's 4 GPU-years total. Probably a large multiple of this was used for
experiments and hyperparameter tuning. I wouldn't be that surprised if overall DeepMind used several hundred GPU-years.
Given that I have only one GPU, I can't replicate everything they've done.
* TensorFlow is great!It has some very nice tutorials on its website, and once you work through a couple it is very easy to use.
I've never used any machine learning library before but found it very easy to get started with TensorFlow. I am 
particularly impressed that it automatically runs on my GPU with literally no extra work from me: I didn't have
to write a single extra line of code to use the GPU. And I got a 75x speedup from using my GTX 980 Ti GPU over
using my CPU.
* I made a cursory attempt at training a value network, but never got one to work very well :(
* Inspired by [this project](https://github.com/jmgilmer/GoCNN) I trained a network to predict final territory
using KGS games that made it all the way to scoring. It seemed to work OK but in the end I found it much less
interesting than the policy network.

### Code

The code is in the `engine/` folder. A short guide:

* `MoveModels.py`: policy network architectures for move prediction. Final one used was `Conv12PosDepELU`.
* `Training.py`: trains the network.
* `TFEngine.py`: runs the trained network to generate moves in a game
* `Features.py`: computes the input features for the network.
* `SGReader.py`: an SGF file parser.
* `GTP.py`: implementation of the Go Text Protocol.

There are a lot of files related to unfinished work on a value network, an "influence" network (to predict final territory), and a tree search engine that would have combined a policy network and a value network.

### Links

In addition to the GoGoD database, I also looked at these databases:

* KGS database of about [180,000 games](http://u-go.net/gamerecords/) 
"where either one of the players (or both) is 7d or stronger, or both are 6d"
* KGS database of about [1.5 million games](http://www.u-go.net/gamerecords-4d) where 
(I think) at least one player is at least 4d.
* I think there are databases of games from the Tygem server, but I'm not sure where.
