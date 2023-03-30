import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils import plot_rewards

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# env, agent, timesteps_per_batch, updates_per_iteration, timesteps_per_batch, lr, gamma, clip, render,
#           render_every, save_freq, seed


class PPO():
    """Interacts with and learns form environment."""

    def __init__(self, env, agent, updates_per_iteration, max_timesteps_per_episode, total_timesteps, timesteps_per_batch, lr, gamma, clip, render,
                    render_every, save_freq, seed, policy_type):
        """Initialize an Agent object.

        Params
        =======
            #### NEED TO FILL THIS OUT
            state_size (int): dimension of each state
            seed (int): random seed
        """

        self.env = env
        self.agent = agent
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.total_timesteps = total_timesteps
        self.updates_per_iteration = updates_per_iteration
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.render = render
        self.render_every = render_every
        self.save_freq = save_freq
        self.seed = seed
        self.policy_type = policy_type

        # Initialize actor and critic networks
        if policy_type == 'CNN':
            self.actor = self.agent.qnetwork_local  # this is the "policy"
            self.critic = self.agent.qnetwork_critic  # this is essentiall a Q function
        elif policy_type == 'BERT':
            self.actor = self.agent.bert_actor
            self.critic = self.agent.bert_critic

        # Initialize optimizers for actor and critic
        self.actor_optim = self.agent.optimizer
        self.critic_optim = self.agent.optimizer

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.agent.num_classes,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var) # 40 x 40

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

    def get_action(self, obs, attention_mask):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action

        obs = torch.from_numpy(obs)
        obs = obs.to(device)
        attention_mask = torch.from_numpy(attention_mask)
        attention_mask = attention_mask.to(device)
        if self.policy_type == 'BERT':
            logits = self.actor.float()(obs, attention_mask)
        else:
            logits = self.actor.float()(obs)
        probs = Categorical(logits=logits)
        action = probs.sample()
        # Calculate the log probability for that action
        log_prob = probs.log_prob(action)
        # Return the sampled action and the log probability of that action in our distribution
        return action.detach(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts, batch_attention_masks):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        if self.policy_type == 'BERT':
            V = self.critic(batch_obs, batch_attention_masks)
            action_probs = self.actor.float()(batch_obs.to(device), batch_attention_masks.to(device))
        else:
            V = self.critic(batch_obs).squeeze()
            action_probs = self.actor.float()(batch_obs.to(device))

        dist = Categorical(logits=action_probs)
        # Calculate the log probability for that action
        log_probs = dist.log_prob(batch_acts.to(device))
        dist_entropy = dist.entropy()

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist_entropy

    def rollout(self):
        """
            we collect the batch of data from simulation.
            we'll need to collect a fresh batch of data each time we iterate the actor/critic networks.
            Parameters:
                None
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_resp = []
        batch_lens = []
        batch_scores = []
        batch_ep_acts = []
        batch_attention_masks = []

        t = 0  # timesteps we've run so far this batch
        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            # Reset the environment. sNote that obs is short for observation. - STATE
            obs, original_response, attention_mask = self.env.reset()
            ep_rews = []  # rewards collected per episode
            ep_resp = [original_response]  # responses collected per episode
            ep_acts = []  # actions chosen per episode

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1  # Increment timesteps ran this batch so far
                batch_obs.append(obs)  # Track observations in this batch
                batch_attention_masks.append(attention_mask) # when using BERT as policy

                # Calculate action and make a step in the env.
                action, log_prob = self.get_action(obs, attention_mask)
                obs, rew, done, response = self.env.step(action.item(), None)

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_acts.append(action.item())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_resp.append(response)

                # If the environment tells us the episode is terminated, break
                if done:
                    break
            batch_resp.append(ep_resp)
            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_scores.append(np.sum(ep_rews))
            batch_ep_acts.append(ep_acts)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_attention_masks = torch.tensor(np.array(batch_attention_masks), dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_resp, \
               batch_scores, batch_ep_acts, batch_attention_masks

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.
            Parameters:
                None
            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.cpu().float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def ppo(self):

        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {self.total_timesteps} timesteps")

        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        avg_rews = []
        scores_list = []
        actions_list = []
        responses_list = []
        while t_so_far < self.total_timesteps:
            # we're collecting our batch simulations here
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, \
                        batch_lens, batch_resp, batch_scores, batch_ep_acts, batch_attention_masks = self.rollout()

            scores_list.append(batch_scores)
            actions_list.append(batch_ep_acts)
            responses_list.append(batch_resp)

            avg_rews.append(torch.mean(batch_rtgs).item())

            t_so_far += np.sum(batch_lens) # Calculate how many timesteps we collected this batch
            i_so_far += 1 # Increment the number of iterations

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            V, old_log_probs, old_dist_entropy = self.evaluate(batch_obs, batch_acts, batch_attention_masks)  # Calculate advantage at k-th iteration
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  # Normalizing advantages

            # update our network for some n epochs
            for _ in range(self.updates_per_iteration):

                V, curr_log_probs, dist_entropy = self.evaluate(batch_obs, batch_acts, batch_attention_masks)  # Calculate V_phi and pi_theta(a_t | s_t)
                ratios = torch.exp(curr_log_probs - batch_log_probs.to(device))  # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)

                A_k = A_k.to(device)
                surr1 = ratios * A_k  # Calculate surrogate losses.
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # the old version here...
                # Calculate actor and critic losses. (min negative -> maximizing)

                # which dist entropy - old or new?!?!
                actor_loss = -(torch.min(surr1, surr2) + 0.01 * dist_entropy).mean()  # added entropy term here

                critic_loss = nn.MSELoss()(V, batch_rtgs + 0.01 * dist_entropy.cpu().detach())
                #critic_loss = nn.MSELoss()(V, batch_rtgs)7
                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()
            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

        datetime = time.strftime("%Y%m%d-%H%M%S")
        responses_unraveled = [item for sublist in responses_list for item in sublist]
        out_df = pd.DataFrame(responses_unraveled)
        scores_unraveled = [item for sublist in scores_list for item in sublist]
        out_df['scores'] = scores_unraveled
        actions_unraveled = [item for sublist in actions_list for item in sublist]
        out_df['actions'] = actions_unraveled
        out_df.to_csv('output/rl_scores/' + datetime + '_ppo.csv')

        plot_rewards(avg_rews, path=None, name='output/rl_scores/' + datetime + '_ppo', rolling_n=100)
