# AlphaGOZero
This is a trial implementation of DeepMind's Oct19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ). 

---

## From Paper:

>Our program, AlphaGo Zero, differs from AlphaGo Fan and AlphaGo Lee 12 in several im- portant aspects. First and foremost, it is trained solely by **self-play reinforcement learning, starting from random play,** without any supervision or use of human data. Second, it only **uses the black and white stones from the board as input features.** Third, it **uses a single neural network, rather than separate policy and value networks.** Finally, it **uses a simpler tree search that relies upon this single neural network to evaluate positions and sample moves, without performing any Monte- Carlo rollouts.** To achieve these results, we introduce a new reinforcement learning algorithm that **incorporates lookahead search inside the training loop,** resulting in ***rapid improvement and precise and stable learning.***
