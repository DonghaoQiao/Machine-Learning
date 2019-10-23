# Multi-Armed Badit with epsilon-greedy
import numpy as np

# number of bandits
k = 3
# action values
Q = np.zeros(k)
# This is to keep track of the number of times we take each action
N = np.zeros(k)
# epsilon value for exploration
eps = 0.1
# iterations
iters=50000
# true probability of winning for each bandit
p_bandits = [0.45, 0.40, 0.80]
def pull(a):
    # pull arm of bandit with index `i` and return 1 if win, else return 0.
    if np.random.rand() < p_bandits[a]:
        return 1
    else:
        return 0
for i in range(iters):
    if np.random.rand() > eps:
        # take greedy action most of the time
        a = np.argmax(Q)
    else:
        # take random action with probability eps
        a = np.random.randint(0, k)
    # collect reward
    reward = pull(a)
    # incremental average
    N[a] += 1
    Q[a] += 1/N[a] * (reward - Q[a])

# Q will converge to the true probability
print(Q)
