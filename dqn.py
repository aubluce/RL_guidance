#  core dqn function

from utils import plot_rewards

import pandas as pd
import numpy as np
import time


def dqn(env, agent, n_episodes=100000, max_t=24, eps_start=1.0, eps_end=0.001, eps_decay=0.999,
        output_path='/output/rl_scores/', plot_rolling_n=300, train_episodes=100000):
    """
    Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time steps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []
    t_list = []
    resp_list = []
    eps = eps_start
    actions = []
    for i_episode in range(1, n_episodes+1):
        if i_episode < train_episodes:
            state, orig_response = env.reset()
        else:
            state, orig_response = env.reset_test()
        episode_scores = []
        episode_responses = [orig_response]
        episode_actions = []
        old_action = None
        for t in range(max_t):
            action_index = agent.act(state, eps)
            episode_actions.append(action_index)
            next_state, reward, done, response = env.step(action_index, old_action)
            agent.step(state, action_index, reward, next_state, done)
            state = next_state
            episode_scores.append(reward)
            episode_responses.append(response)
            if done:
                break
            eps = max(eps*eps_decay, eps_end)
            old_action = action_index
        if i_episode % 500 == 0:
            print(i_episode, np.sum(episode_scores), eps)
        resp_list.append(episode_responses)
        if i_episode % 1000 == 0:
            pd.DataFrame(scores).to_csv(output_path + 'partial_scores.csv')
            pd.DataFrame(resp_list).to_csv(output_path + 'partial_response_list.csv')
        scores.append(np.sum(episode_scores))
        t_list.append(t)
        actions.append(episode_actions)
    datetime = time.strftime("%Y%m%d-%H%M%S")
    out_df = pd.DataFrame(resp_list)
    out_df['scores'] = scores
    out_df['actions'] = actions
    out_df.to_csv(output_path + datetime + '.csv')

    return plot_rewards(scores, path=None,  name=output_path + datetime, rolling_n=plot_rolling_n)


