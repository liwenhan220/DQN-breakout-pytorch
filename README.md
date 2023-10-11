# DQN-breakout-pytorch
I recovered my better version of DQN from the older version

# Working requirements
gym==0.17.0, gym[atari], pytorch, matplotlib

# Train your model
run `python run.py` will train a model from fresh start

# Check performance of a model
use `python eval.py` to check the performance of a model. Don't confuse `eval.py` with `new_eval.py`, because `new_eval.py` is for evaluation during training. I gave it a bad name.

# Viewing hightlight moments
During the training process, every improvement (higher rewards) will be recorded inside the `vlog` folder. Use `python view_log.py` to convert files inside to videos. I have converted some and stored them in the `video` folder, check it out. The highest reward achieved was 428.

# Pretrained models
I saved a model every time its evaluation reward improves. I only uploaded the best model (in terms of evaluation reward) and the most recent model here.
