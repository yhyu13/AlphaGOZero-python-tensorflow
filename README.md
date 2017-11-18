# AlphaGOZero (python tensorflow implementation)
This is a trial implementation of DeepMind's Oct19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).

>**This repository has single purpose of education only**

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

[repo: leela-zero (c++ popular 9dan Go A.I. owned by Mozilla)](https://github.com/gcp/leela-zero)

[repo: reversi-alpha-zero (if you like reversi(黑白棋))](https://github.com/mokemokechicken/reversi-alpha-zero)

[repo: Sabaki](https://github.com/yishn/Sabaki/releases)

---

# Set up

## Install requirement

python 3.6

```
pip install -r requirement.txt
```

## Download Dataset (kgs 4dan)

Under repo's root dir

```
cd data/download
chmod +x download.sh
./download.sh
```

## Preprocess Data

*It is only an example, feel free to assign your local dataset directory*

```
python preprocess.py preprocess ./data/SGFs/kgs-*
```

## Train A Model

```
python main.py --mode=train --force_save —-n_resid_units=20
```

## Play Against An A.I. (currently only random A.I. is available)

```
python main.py --mode=gtp —-policy=random

2017-11-16 02:19:45,274 [20046] DEBUG    Network: Building Model Complete...Total parameters: 1581959
2017-11-16 02:19:45,606 [20046] DEBUG    Network: Loading Model...
2017-11-16 02:19:45,615 [20046] DEBUG    Network: Loading Model Failed
2017-11-16 02:19:46,702 [20046] DEBUG    Network: Done initializing variables
GTP engine ready
clear_board
=


showboard
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . . 19
18 . . . . . . . . . . . . . . . . . . . 18
17 . . . . . . . . . . . . . . . . . . . 17
16 . . . . . . . . . . . . . . . . . . . 16
15 . . . . . . . . . . . . . . . . . . . 15
14 . . . . . . . . . . . . . . . . . . . 14
13 . . . . . . . . . . . . . . . . . . . 13
12 . . . . . . . . . . . . . . . . . . . 12
11 . . . . . . . . . . . . . . . . . . . 11
10 . . . . . . . . . . . . . . . . . . . 10
 9 . . . . . . . . . . . . . . . . . . .  9
 8 . . . . . . . . . . . . . . . . . . .  8
 7 . . . . . . . . . . . . . . . . . . .  7
 6 . . . . . . . . . . . . . . . . . . .  6
 5 . . . . . . . . . . . . . . . . . . .  5
 4 . . . . . . . . . . . . . . . . . . .  4
 3 . . . . . . . . . . . . . . . . . . .  3
 2 . . . . . . . . . . . . . . . . . . .  2
 1 . . . . . . . . . . . . . . . . . . .  1
   A B C D E F G H J K L M N O P Q R S T
Move: 0. Captures X: 0 O: 0

None
=

play Black B1
= (1, (2, 1))

```

## Play in Sabaki

The support for Sabaki is **available**. Go to [Sabaki-releases](https://github.com/yishn/Sabaki/releases) and grab the latest version for your device.

1. Open Sabaki, play around with **'View'** and checkout **'Engine'**. Go to 'Manage engine' and add ```/path/to/main.py``` to 'path' with argument ```--mode=gtp```.
2. Open ```/path/to/main.py``` with your IDE/CMD, etc. Change the bash bang command (i.e. ```#!/usr/.../python```) to the output of ```which python``` on your device.
3. Go back to Sabaki, checkout **'Engine'** and toggle **'Attach..'**. Select A.I. as either ```W``` or ```B```, or both.
4. Click Ok. All set!
5. By the same token, add GNU Go and Leela under ```engine``` folder to Sabaki as well!

![](/figure/Sabaki.png)

>**The mini AlphaGo Zero (6 layer) trained under KGS dataset fall short against GUN Go by using the policy network only (without MCTS)**

**My model still has strange bug when evaluating the root node. As you can see in console above. It outputs the same result again and again.**

## Fully functional Self Play Pipeline

Under repo’s root  dir

```
python main.py --mode=selfplay


2017-11-17 22:18:57,857 [35771] DEBUG    Network: Building Model...
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:58,258 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:58,465 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:58,668 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:58,879 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:59,082 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:59,315 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
2017-11-17 22:19:05,152 [35771] DEBUG    Network: Building Model Complete...Total parameters: 7505819
2017-11-17 22:19:09,631 [35771] DEBUG    Network: Done initializing variables
2017-11-17 22:19:09,631 [35771] DEBUG    Network: Loading Model...
INFO:tensorflow:Restoring parameters from ./savedmodels/model-0.4114.ckpt-347136
2017-11-17 22:19:09,639 [35771] INFO     tensorflow: Restoring parameters from ./savedmodels/model-0.4114.ckpt-347136
2017-11-17 22:19:09,962 [35771] DEBUG    Network: Loading Model Succeeded...
...
...
...
2017-11-17 22:18:56,122 [35771] DEBUG    model.APV_MCTS_C: Greedy move is: (2, 3) with prob 0.039
2017-11-17 22:18:56,123 [35771] INFO     model.APV_MCTS_C: Win rate for player W is 0.5000
2017-11-17 22:18:56,123 [35771] DEBUG    model.APV_MCTS_C: Move at step 436 is (12, 14)
2017-11-17 22:18:56,305 [35771] DEBUG    model.APV_MCTS_C: Greedy move is: (18, 5) with prob 0.064
2017-11-17 22:18:56,307 [35771] INFO     model.APV_MCTS_C: Win rate for player B is 0.5000
2017-11-17 22:18:56,307 [35771] DEBUG    model.APV_MCTS_C: Move at step 437 is None
2017-11-17 22:18:56,495 [35771] DEBUG    model.APV_MCTS_C: Greedy move is: (2, 3) with prob 0.047
2017-11-17 22:18:56,497 [35771] INFO     model.APV_MCTS_C: Win rate for player W is 0.5000
2017-11-17 22:18:56,498 [35771] DEBUG    model.APV_MCTS_C: Move at step 438 is None
2017-11-17 22:18:56,499 [35771] DEBUG    model.SelfPlayWorker: Game #1 Final Position:
   A B C D E F G H J K L M N O P Q R S T
19 O O X . X O O O . O O O . O X X . X . 19
18 O X X X X X O O O O O O O O O X X . X 18
17 O O X . X X X O X O X O X O X X . X O 17
16 . O O X X X O O X X X O X O X . X O O 16
15 O O X X O X O O X O X X X O X X O O O 15
14 O O O O O O O X X O O X O O O X O O . 14
13 O X O O O O O O X O O X X X O O . O O 13
12 X X X X O O O X O O O O O X X O O O X 12
11 . X X O O . O X X O . O . O X X O X X 11
10 X X X O O O X X X X O O O O O X X X . 10
9 X X . X O X X X . X O . O O X X X . X  9
8 X . X X O O X . X . X O . O O O O X .  8
7 X X . X X O O X X X X O O O O O X . X  7
6 X X X X . X X . X X X X O O . O X X X  6
5 X X X . X . X X X X X X O O O O O O O  5
4 . X X X X X X X X . X X X O . O X X X  4
3 X . X X X X . X X X . X X X O O X X X  3
2 . X X X . X X X X X X . X X X O X X .  2
1 X . X X X . X X X . X X . X O O X . X  1
  A B C D E F G H J K L M N O P Q R S T
Move: 438. Captures X: 86 O: 36

2017-11-17 22:18:56,500 [35771] INFO     model.SelfPlayWorker: Self-Play Simulation Game #1: 80.351 seconds
2017-11-17 22:18:56,501 [35771] DEBUG    Network: Running evaluation...
2017-11-17 22:18:57,849 [35771] DEBUG    Network: Test loss: -0.00
2017-11-17 22:18:57,849 [35771] DEBUG    Network: Play move test accuracy: -0.0000
2017-11-17 22:18:57,849 [35771] DEBUG    Network: Win ratio test accuracy: -0.00
2017-11-17 22:18:57,852 [35771] INFO     model.SelfPlayWorker: test set evaluation: 1.352 seconds
2017-11-17 22:18:57,857 [35771] DEBUG    Network: Building Model...
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:58,258 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:58,465 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:58,668 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:58,879 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:59,082 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
INFO:tensorflow:image after unit (?, 19, 19, 256)
2017-11-17 22:18:59,315 [35771] INFO     tensorflow: image after unit (?, 19, 19, 256)
2017-11-17 22:19:05,152 [35771] DEBUG    Network: Building Model Complete...Total parameters: 7505819
2017-11-17 22:19:09,631 [35771] DEBUG    Network: Done initializing variables
2017-11-17 22:19:09,631 [35771] DEBUG    Network: Loading Model...
INFO:tensorflow:Restoring parameters from ./savedmodels/model-0.4114.ckpt-347136
2017-11-17 22:19:09,639 [35771] INFO     tensorflow: Restoring parameters from ./savedmodels/model-0.4114.ckpt-347136
2017-11-17 22:19:09,962 [35771] DEBUG    Network: Loading Model Succeeded...
2017-11-17 22:20:06,836 [35771] DEBUG    model.APV_MCTS_C: Model evaluation game results : ['W+48.5']
2017-11-17 22:20:06,836 [35771] INFO     model.SelfPlayWorker: Previous Generation win by 0.0000% the game! 姜还是老得辣!
2017-11-17 22:20:06,836 [35771] INFO     Network: NETWORK SHUTDOWN!!!
2017-11-17 22:20:06,837 [35771] INFO     __main__: Global epoch 0 finish.
```

> **Notice the self play against best model would allocate memory to build another network. More thoughtful version could be initialize multiple models in a single tensorflow graph.**

## Nov 15th Supervised Learning result

The Supervised Learning is done on a 6 layer deep neural net which has the same architecture. It is trained on 2016-2017 KGD4-dan dataset, about 250,000 games for 5 epochs and is evaluated on 100,000 positions. It achieve 40% professional move prediction accuracy on the evaluation dataset consistently. Strangely, the game result prediction error (MSE) stays around 1 consistently, in contrast to what DeepMind's graph where the error never goes beyond 1.

![](/figure/Nov15acc.png)
![](/figure/Nov15ce.png)
![](/figure/Nov15mse.png)

### DeepMind's training result:

![](/figure/rl_vs_sl.png)

---

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
* 1. A convolution of 2 filters of kernel size 1 x 1 with stride 1
  2. Batch normalisation
  3. A rectifier non-linearity
  4. A fully connected linear layer that outputs a vector of size 192^2 + 1 = 362 corresponding to logit probabilities for all intersections and the pass move

**Value Head**
* 1. A convolution of 1 filter of kernel size 1 x 1 with stride 1
  2. Batch normalisation
  3. A rectifier non-linearity
  4. A fully connected linear layer to a hidden layer of size 256
  5. A rectifier non-linearity
  6. A fully connected linear layer to a scalar
  7. A tanh non-linearity outputting a scalar in the range [ 1, 1]

---

# TODO:

- [x] AlphaGo Zero Architecture
- [x] Supervised Training
- [x] Self Play pipeline
- [x] Go Text Protocol
- [x] Sabaki Engine enabled
- [ ] *Tabula rasa*
- [ ] Keras implementation
- [ ] Distributed learning

# Credit:

*Brain Lee
*Ritchie Ng
