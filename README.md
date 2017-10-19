# AlphaGOZero
This is a trial implementation of DeepMind's Oct19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ). 

---

## From Paper:

>Our program, AlphaGo Zero, differs from AlphaGo Fan and AlphaGo Lee 12 in several im- portant aspects. First and foremost, it is trained solely by **self-play reinforcement learning, starting from random play,** without any supervision or use of human data. Second, it only **uses the black and white stones from the board as input features.** Third, it **uses a single neural network, rather than separate policy and value networks.** Finally, it **uses a simpler tree search that relies upon this single neural network to evaluate positions and sample moves, without performing any Monte- Carlo rollouts.** To achieve these results, we introduce a new reinforcement learning algorithm that **incorporates lookahead search inside the training loop,** resulting in ***rapid improvement and precise and stable learning.***


Congratulation to DeepMind to pierce the frontier once again! AlphaGO Zero (fully self-play by reinforcement learning with no human games examples).

I downloaded the paper Mastering the Game of Go without Human Knowledge in the first place, but only found myself lack prior knowledge in Monte Carlo Search Tree (MCST). I tried my best to highlight what is interesting.

This time's AlphaGo uses combined policy & value network (final fc diverges to two branches) to cope with training stability. 
![](/figure/dual_network.png)
Innovation (annealing & Dirichlet noise) in MCTS has enabled exploration 
![](/figure/MCTS.png)
And exploration leads to learning more and more complex movings, making the game at the end of training (~70h) both competitive and balanced.
![](/figure/learning_go.png)
The input is still raw stones but normal CNN has been replaced by RES-50NET
![](/figure/cnn_archi.png)
And finally pure RL has outperformed supervised learning+RL agent
![](/figure/rl_vs_sl.png)