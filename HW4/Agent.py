import random
class Agent:
    def __init__(self, k, epsilon, alpha = None):
        self.k = k
        self.epsilon = epsilon
        self.cnts = [] # choose time
        self.qs = [] # Q_values
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.cnts = [0 for i in range(self.k)]
        self.qs = [0.0 for i in range(self.k)]

    def select_action(self):
        rnd_val = random.random()
        if rnd_val < self.epsilon:
            return random.randrange(self.k)
        max_q = max(self.qs)
        candiates = []
        for i, q in enumerate(self.qs):
            if q == max_q:
                candiates.append(i)
        return random.choice(candiates)
    
    def update_q(self, action, reward):
        if self.alpha is None:
            self.cnts[action] += 1
            now_cnt = self.cnts[action]
            q_val = self.qs[action]
            self.qs[action] += (1.0 / now_cnt) * (reward - q_val)
        else:
            q_val = self.qs[action]
            self.qs[action] += self.alpha * (reward - q_val)