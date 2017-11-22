# AlphaGOZero (python tensorflow implementation)
This is a trial implementation of DeepMind's Oct19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).

>**This repository has single purpose of education only**

---
# From Paper

Pure RL has outperformed supervised learning+RL agent

![](/figure/rl_vs_sl.png)

---

# Set up

## Install requirement

python 2.7

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

- [x] AlphaGo Zero Architecture
- [x] Supervised Training
- [x] Self Play pipeline
- [x] Go Text Protocol
- [x] Sabaki Engine enabled
- [ ] *Tabula rasa*
- [ ] Keras implementation
- [ ] Distributed learning

# Supervised Learning result (11/8/2017)

## Precondition

Dataset:

> * Train: 65536*11 samples
> * Test: 100000 samples

Model:

> AlphaGOzero 20 block elu variation

Server:

> AWS P3 8xlarge

## Training move prediction

![Screen Shot 2017-11-08 at 10.14.49 AM.png](http://upload-images.jianshu.io/upload_images/1873837-f298f9760f8c9bb4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

(*Steps refer to mini-batch updates, each mini-batch has 2048 samples*)

## Training Total Loss (1*CE + 0.01*MSE)

![Screen Shot 2017-11-08 at 10.14.31 AM.png](http://upload-images.jianshu.io/upload_images/1873837-3d98dae9280e22eb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

(*Steps refer to mini-batch updates, each mini-batch has 2048 samples*)

## Remark

1. Training acc > 70%, but evaluation acc < 6%. Therefore, no model is saved.
2. Need code review, presumably use batch norm incorrectly.
3. Validate covergence of supervised learning, and the training accuracy proposed by DeepMind
4. Total training time: 7h 12m 47s

## TODO

* Record CE and MSE separately. (Done)
* Find error that causes ineffective evaluation on the test dataset. (Done)
* Retrain, and qunatize saved models for fast inference.
* Open trained model to the world.

# Credit:

*Brain Lee
*Ritchie Ng
