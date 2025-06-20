import random
class BanditEnv:
    def __init__(self, k, stationary = False):
        self.k = k
        self.action_history = []
        self.reward_history = []
        self.means = [random.gauss(0, 1) for i in range(self.k)]
        self.stationary = stationary
    
    def reset(self):
        self.action_history.clear()
        self.reward_history.clear()
    
    def random_walk(self):
        for i in range(self.k):
            self.means[i] += random.gauss(0, 0.01)

    def step(self, action: int):
        reward = random.gauss(self.means[action], 1)
        self.action_history.append(action)
        self.reward_history.append(reward)
        if self.stationary == False:
            self.random_walk()
        return reward
    def export_history(self):
        return self.action_history, self.reward_history