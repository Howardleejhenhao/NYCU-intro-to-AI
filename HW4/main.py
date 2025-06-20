import matplotlib.pylab as plt
from BanditEnv import BanditEnv
from Agent import Agent
from tqdm import tqdm
import numpy as np
def run_exper(k, eps, run_time, step, stationary = True, alpha = None):
    r_sm = [0.0] * step
    opt_cnt = [0] * step
    for run in tqdm(range(run_time), desc = f"eps = {eps}"):
        banditenv = BanditEnv(k, stationary)
        banditenv.reset()
        agent = Agent(k, eps, alpha = alpha)
        agent.reset()
        for i in range(step):
            act = agent.select_action()
            reward = banditenv.step(act)
            agent.update_q(act, reward)
            current_opt = max(range(k), key=lambda j: banditenv.means[j])
            r_sm[i] += reward
            if act == current_opt:
                opt_cnt[i] += 1

    ret_avg_reward = [total / run_time for total in r_sm]
    ret_opt_pre = [100.0 * (count / run_time) for count in opt_cnt]
    return ret_avg_reward, ret_opt_pre

def plt_res(result, run_time):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for eps, (avg_r, _) in result.items():
        plt.plot(np.arange(1, run_time+1), avg_r, label=f"ε = {eps}")
    plt.xlabel("Time step")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("Average Reward over Time")
    
    plt.subplot(1, 2, 2)
    for eps, (_, pct_opt) in result.items():
        plt.plot(np.arange(1, run_time+1), pct_opt, label=f"ε = {eps}")
    plt.xlabel("Time step")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.title("Optimal Action % over Time")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Part 3
    k = 10
    run_time = 2000
    step = 1000
    epsilon = [0.0, 0.1, 0.01]
    result = {}
    for eps in epsilon:
        r_avg, opt_pre  = run_exper(k, eps, run_time, step)
        result[eps] = (r_avg, opt_pre)
    plt_res(result, step)

    # Part 5
    k = 10
    run_time = 2000
    step = 10000
    epsilon = [0.0, 0.1, 0.01]
    result = {}
    for eps in epsilon:
        r_avg, opt_pre  = run_exper(k, eps, run_time, step, stationary=False)
        result[eps] = (r_avg, opt_pre)
    plt_res(result, step)

    # Part 7
    k = 10
    run_time = 2000
    step = 10000
    epsilon = [0.0, 0.1, 0.01]
    result = {}
    alpha = 0.1
    for eps in epsilon:
        r_avg, opt_pre  = run_exper(k, eps, run_time, step, stationary = False, alpha = alpha)
        result[eps] = (r_avg, opt_pre)
    plt_res(result, step)

    # Discussion
    # k = 10
    # run_time = 2000
    # step = 20000

    # epsilons = [0.0, 0.01, 0.1]
    # alphas   = [None, 0.05, 0.1]

    # result = {}
    # for eps in epsilons:
    #     for alpha in alphas:
    #         r_avg, opt_pre = run_exper(
    #             k, eps, run_time, step,
    #             stationary=False,
    #             alpha=alpha
    #         )
    #         label = f"ε={eps}, α={alpha}"
    #         result[label] = (r_avg, opt_pre)

    # plt_res(result, step)
