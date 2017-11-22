# AlphaGOZero (python tensorflow implementation)
This is a trial implementation of DeepMind's Oct19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).

>**This repository has single purpose of education only**

---
# From Paper

Pure RL has outperformed supervised learning+RL agent

![](/figure/rl_vs_sl.png)


# Nov20 SL evaluation

![](/figure/Nov20large20eval.png)

## Download trained model

1. [https://drive.google.com/drive/folders/1Xs8Ly3wjMmXjH2agrz25Zv2e5-yqQKaP?usp=sharing](https://drive.google.com/drive/folders/1Xs8Ly3wjMmXjH2agrz25Zv2e5-yqQKaP?usp=sharing)

2. Place under ./savedmodels/large20/

---

# Set up

## Install requirement

python 3.6
tensorflow/tensorflow-gpu

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
python main.py --mode=train
```

## Play Against An A.I. (currently only random A.I. is available)

```
python main.py --mode=gtp —-policy=randompolicy --model_path='./savedmodels/model--0.0.ckpt'
```

## Play in Sabaki

![](/figure/Sabaki.png)

1. In console:
```
which python
```
add result to the headline of ```main.py``` with ```#!``` prefix.

2. Add the path of ```main.py``` to Sabaki's manage Engine with argument ```--mode=gtp```

# TODO:
- [x] AlphaGo Zero Architecture
- [ ] Supervised Training
- [x] Self Play pipeline
- [x] Go Text Protocol
- [x] Sabaki Engine enabled
- [ ] *Tabula rasa*
- [ ] Keras implementation
- [ ] Distributed learning

# Credit:

*Brain Lee
*Ritchie Ng
*Samuel Graván
*森下 健
