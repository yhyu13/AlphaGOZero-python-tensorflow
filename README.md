# AlphaGOZero
This is a trial implementation of DeepMind's Oct19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ). 

---

## Useful links:

[All DeepMind’s AlphaGO games](http://www.alphago-games.com)

[GoGOD dataset, $15](https://gogodonline.co.uk)

[KGS >=4dan, FREE](https://www.u-go.net/gamerecords-4d/)

[Youtube: Learn to play GO](https://www.youtube.com/watch?v=xMshtO8h7RU)

[repo: MuGo](https://github.com/brilee/MuGo)

[repo: ROCAlphaGO](https://github.com/Rochester-NRT/RocAlphaGo)

[repo: miniAlphaGO](https://github.com/yotamish/mini-Alpha-Go)

[repo: resnet-tensorflow](https://github.com/ritchieng/resnet-tensorflow)

[repo: leela-zero (best AlphaGo Zero replica so far)](https://github.com/gcp/leela-zero)

## From Paper:

>Our program, AlphaGo Zero, differs from AlphaGo Fan and AlphaGo Lee 12 in several im- portant aspects. First and foremost, it is trained solely by **self-play reinforcement learning, starting from random play,** without any supervision or use of human data. Second, it only **uses the black and white stones from the board as input features.** Third, it **uses a single neural network, rather than separate policy and value networks.** Finally, it **uses a simpler tree search that relies upon this single neural network to evaluate positions and sample moves, without performing any Monte- Carlo rollouts.** To achieve these results, we introduce a new reinforcement learning algorithm that **incorporates lookahead search inside the training loop,** resulting in ***rapid improvement and precise and stable learning.***


Congratulation to DeepMind to pierce the frontier once again! AlphaGO Zero (fully self-play by reinforcement learning with no human games examples).

I downloaded the paper Mastering the Game of Go without Human Knowledge in the first place, but only found myself lack prior knowledge in Monte Carlo Search Tree (MCST). I tried my best to highlight what is interesting.

This time's AlphaGo uses combined policy & value network (final fc diverges to two branches) to cope with training stability.
From Paper:

![](/figure/dual_network.png)

Innovation (annealing & Dirichlet noise) in MCTS has enabled exploration

From Paper:
![](/figure/MCTS.png)

And exploration leads to learning more and more complex movings, making the game at the end of training (~70h) both competitive and balanced.

From Paper:
![](/figure/learning_go.png)

The input is still raw stones but normal CNN has been replaced by residual net

From Paper:
![](/figure/cnn_archi.png)

And finally pure RL has outperformed supervised learning+RL agent

From Paper:
![](/figure/rl_vs_sl.png)

## AlphaGo Zero Architecture:

* input 19 x 19 x 17: 7 previous states + current state player’s stone, 7 previous states + current state opponent’s stone, player’s colour
* 1. A convolution of 256 filters of kernel size 3 x 3 with stride 1
  2. Batch normalisation
  3. A rectifier non-linearity

**Residual Blocks**
* 1. A convolution of 256 filters of kernel size 3 x 3 with stride 1
  2. Batch normalisation
  3. A rectifier non-linearity
  4. A convolution of 256 filters of kernel size 3 x 3 with stride 1
  5. Batch normalisation
  6. A skip connection that adds the input to the block
  7. A rectifier non-linearity

**Policy Head**
* 1.A convolution of 2 filters of kernel size 1 x 1 with stride 1  2. Batch normalisation  3. A rectifier non-linearity  4. A fully connected linear layer that outputs a vector of size 192^2 + 1 = 362 corresponding to logit probabilities for all intersections and the pass move

**Value Head**
* 1. A convolution of 1 filter of kernel size 1 x 1 with stride 1 
  2. Batch normalisation  3. A rectifier non-linearity
  4. A fully connected linear layer to a hidden layer of size 256 
  5. A rectifier non-linearity  6. A fully connected linear layer to a scalar  7. A tanh non-linearity outputting a scalar in the range [ 1, 1]

---

# Set up

## Install requirement

python 3.6

```
pip install -r requirement.txt
```

## Add ./util ./model to PYTHONPATH

Under repo's root dir

```
PWD=pwd
export PYTHONPATH="$PWD/model:$PYTHONPATH"
export PYTHONPATH="$PWD/utils:$PYTHONPATH"
set PWD=
```

## Download Dataset (kgs 4dan):

Under repo's root dir

```
cd data/download
chmod +x download.sh
./download.sh
```

## Preprocess Data:

*It is only an example, feel free to assign your local dataset directory*

```
python preprocess.py --dataset=./data
```

## Train A Model:

```
python train.py
```

# Credit:

*Brain Lee
*Ritchie Ng
