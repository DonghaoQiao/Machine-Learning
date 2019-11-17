# Introduction of Reinforcement Learning  

[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)  
[Reinforcement Learning: An Introduction中文版](https://rl.qiwihui.com/zh_CN/latest/index.html)  
[UC Berkeley Video](https://www.youtube.com/watch?v=Q4kF8sfggoI&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3)  
[DeepMind Video](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)  
  
* [Multi-Armed Bandit](https://github.com/DonghaoQiao/Machine-Learning/blob/master/Reinforcement%20Learning/Multi_Armed_Bandit.py)  
* Markov Decision Processes (MDP)  
* Monte Carlo Methods  
* Temporal-Difference Learning  
* Sarsa  
* Q-learning  

# [GYM](https://gym.openai.com)
Gym is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like Pong or Pinball.  
The environment’s step function returns exactly what we need. In fact, step returns four values. These are:

observation (object): an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
reward (float): amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment’s last state change). However, official evaluations of your agent are not allowed to use this for learning.
* [Gym Hello World](https://github.com/DonghaoQiao/Machine-Learning/blob/master/Reinforcement%20Learning/Gym_CartPole.py)  
