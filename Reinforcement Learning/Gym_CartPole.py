import gym

env = gym.make('CartPole-v0')
print('observation size: ', env.observation_space)
#> observation size:  Box(4,)
print(env.observation_space.high)
print(env.observation_space.low)
print('action size: ', env.action_space
#> action size: Discrete(2)

for _ in range(10):
    observation = env.reset()
    print(observation)
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
env.close()
