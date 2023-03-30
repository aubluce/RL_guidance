#  rl functions

import numpy as np
import re
from collections import deque, namedtuple
from scipy.stats import percentileofscore
import random
import nltk
import math

import torch
from torch import optim


from qnets import QNET_CNN, QNetwork
from bert_policies import BERTPolicy

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size 100,000
BATCH_SIZE = 16  # minibatch size was 32
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters was -3
LR = 5e-3  # learning rate was -4
UPDATE_EVERY = 4  # how often to update the network was 4


class ReplayBuffer:
    """Fixed -size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        states = torch.from_numpy(np.rollaxis(np.dstack([e.state for e in experiences if e is not None]), -1)).float().\
            to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.rollaxis(np.dstack([e.next_state for e in experiences if e is not None]), -1))\
            .float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Environment(object):
    state = torch.Tensor()
    previous_prob = 0
    response = str()
    new_response = str()
    attention_mask = None

    def __init__(self, response_df, test_df, prob_threshold, predictor,
                 w2v_model, glove_model, phrases_list, key_phrase_list, old_action,
                 bert_tokenizer, policy_type):
        self.response_df = response_df
        self.test_df = test_df
        self.prob_threshold = prob_threshold
        self.predictor = predictor
        self.w2v_model = w2v_model
        self.glove_model = glove_model
        self.key_phrase_list = key_phrase_list
        self.phrases_list = phrases_list
        self.predictor = predictor
        self.new_prediction = None
        self.old_prediction = None
        self.action_index = None
        self.old_action = old_action
        self.bert_tokenizer = bert_tokenizer
        self.policy_type = policy_type

    def init_state(self):
        """ Initialize state and calculate initial predicted probabilities """
        self.response_df = self.response_df.copy()
        self.response = self.response_df.sample()['text'].item()
        self.old_prediction = self.get_prediction(self.response)
        if self.policy_type == 'PPO':
            self.state = self.create_state_cnn(self.response)
        elif self.policy_type == 'BERT':
            self.state, self.attention_mask = self.create_state_bert(self.response)

    def init_state_test(self):
        """ Initialize state and calculate initial predicted probabilities """
        self.test_df = self.test_df.copy()
        self.response = self.test_df.sample()['text'].item()
        self.old_prediction = self.get_prediction(self.response)
        if self.policy_type == 'PPO':
            self.state = self.create_state_cnn(self.response)
        elif self.policy_type == 'BERT':
            self.state, self.attention_mask = self.create_state_bert(self.response)

    def get_prediction(self, new_response):
        """ new_response should be a string """
        new_prediction = self.predictor.predict(new_response)  # removed '6. ' + from beg
        new_prediction.sort()
        return [x[1] for x in new_prediction]

    def create_phrase_vector(self, response, sim_threshold=.75):
        response = re.sub("[^0-9a-zA-Z]+", " ", response).lower()

        resp_split = [x.lower() for x in
                      response.replace('.', ' ').replace(',', ' ').replace('-', ' ').replace('6. ', '').replace('  ',
                      ' ').split(' ') if x not in ['', '.', ',']]
        text_match_list = [x in response for x in self.key_phrase_list]  # len same as key phrases (10)

        for i in range(len(self.key_phrase_list)):
            if not text_match_list[i]:
                phrase = self.key_phrase_list[i]
                phrase_len = len(phrase.split(' '))
                ngram_response = [' '.join(x) for x in list(nltk.ngrams(resp_split, phrase_len))]

                for ngram in ngram_response:
                    sim_list = []
                    for j in range(phrase_len):
                        if ngram.split(' ')[j] in self.glove_model.key_to_index:
                            sim_list.append(self.glove_model.similarity(ngram.split(' ')[j], phrase.split(' ')[j]))
                    if len(sim_list) > 0:
                        if np.mean(sim_list) > sim_threshold:
                            text_match_list[i] = True
                            break
        return np.array([int(x) for x in text_match_list])

    def create_state(self, response):
        """ Turn response into matrix of w2v vectors """

        phrase_vector = self.create_phrase_vector(response)
        response = re.sub("[^0-9a-zA-Z]+", " ", response)
        resp_split = [x.lower() for x in
                      response.replace('.', ' ').replace(',', ' ').replace('-', ' ').replace('6. ', '').replace('  ',
                      ' ').split(' ') if x not in ['', '.', ',']]
        if len(resp_split) > 0:
            unpadded_state = self.w2v_model.wv[resp_split].flatten() # without flatten, returns [#words, w2v size]
        else:
            unpadded_state = np.array([0])
        unpadded_state = unpadded_state[:399]
        state = np.pad(unpadded_state, pad_width=(0, 400 - unpadded_state.shape[0]))
        phrase_vector = phrase_vector.astype('float32')
        return np.concatenate([phrase_vector, state])

    def create_state_cnn(self, response):
        """ Turn response into 2D matrix of w2v vectors """
        phrase_vector = self.create_phrase_vector(response)
        phrase_vector = phrase_vector.astype('float32')
        phrase_vector2 = phrase_vector[np.newaxis, :]

        response = re.sub("[^0-9a-zA-Z]+", " ", response)
        resp_split = [x.lower() for x in
                      response.replace('.', ' ').replace(',', ' ').replace('-', ' ').replace('6. ', '').replace('  ',
                      ' ').split(' ') if x not in ['', '.', ',']]

        if len(resp_split) > 0:
            unpadded_state = self.w2v_model.wv[resp_split]  # returns [#words, w2v size]
        else:
            unpadded_state = np.zeros((50, 10))  # this shouldn't be hard coded
        unpadded_state = unpadded_state[:50]  # cap at 50 words/tokens

        state = np.pad(unpadded_state, pad_width=((0, 50-unpadded_state.shape[0]), (0, 0)))  # size 50, 10
        #return state
        # think I need to pad the phrase vector2
        phrase_vector_padded = np.pad(phrase_vector.T, pad_width=((0, 50-phrase_vector2.shape[1])))
        phrase_vector_padded = phrase_vector_padded[np.newaxis, :]
        return np.c_[phrase_vector_padded.T, state]

    def create_state_bert(self, response):
        """

        :param response: student response
        :return: bert tokenized response
        """
        response = re.sub("[^0-9a-zA-Z]+", " ", response)
        bert_input = self.bert_tokenizer(response, padding='max_length', max_length=512,
                                         truncation=True, return_tensors="pt")

        return bert_input['input_ids'].squeeze(1).numpy(), bert_input['attention_mask'].squeeze(1).numpy()

    def take_action3(self, action_index):
        """
        0-101 add phrases
        102 - 112 delete chunk

        :param action_index:
        :return:
        """
        self.response = re.sub('[^0-9a-zA-Z]+', ' ', self.response)
        resp_split = [x.lower() for x in self.response.split(' ') if x not in ['', '.', ',']]
        len_resp = len(resp_split)

        if action_index < 99:  # adding a phrase

            phrase_num = math.ceil((action_index+1)/3) - 1
            phrase_to_add = self.phrases_list[phrase_num]
            if action_index % 3 == 0:
                new_resp = [phrase_to_add] + resp_split
            elif action_index % 3 == 1:
                insert_index = int(len_resp/2)
                new_resp = resp_split[:insert_index] + [phrase_to_add] + resp_split[insert_index:]
            else:
                new_resp = resp_split + [phrase_to_add]

        else:  # deleting a phrase
            if len_resp <= 10:
                del_len = 1
            else:
                del_len = math.ceil(len_resp/10)

            action_percentile = percentileofscore([x for x in range(10)], action_index - 30)
            beg_index = int(action_percentile/100 * len_resp)

            if beg_index + del_len > len_resp:
                end_index = len_resp-1
            else:
                end_index = beg_index + del_len

            new_resp = resp_split[:beg_index] + resp_split[end_index:]
        if self.policy_type == 'PPO':
            self.state = self.create_state_cnn(' '.join(new_resp).strip('.'))
        elif self.policy_type == 'BERT':
            self.state, self.attention_mask = self.create_state_bert(' '.join(new_resp).strip('.'))
        self.response = ' '.join(new_resp).strip()

    def take_action2(self, action_index):
        """
        0-2: add phrase 1
        3-5: add phrase 2
        6-8: add phrase 3
        9-11: add phrase 4
        12-14: add phrase 5
        15-17: add phrase 6
        18-20: add phrase 7
        21-23: add phrase 8
        24-26: add phrase 9
        27-29: add phrase 10
        30-39: delete 1/10th of the response
        :param action_index: should be a number from 0-39
        :return: Non

        """

        self.response = re.sub('[^0-9a-zA-Z]+', ' ', self.response)
        resp_split = [x.lower() for x in self.response.split(' ') if x not in ['', '.', ',']]
        len_resp = len(resp_split)

        if action_index < 30:  # adding a phrase
            phrase_num = math.ceil((action_index+1)/3) - 1
            phrase_to_add = self.phrases_list[phrase_num]
            if action_index % 3 == 0:
                new_resp = [phrase_to_add] + resp_split
            elif action_index % 3 == 1:
                insert_index = int(len_resp/2)
                new_resp = resp_split[:insert_index] + [phrase_to_add] + resp_split[insert_index:]
            else:
                new_resp = resp_split + [phrase_to_add]

        else:  # deleting a phrase
            if len_resp <= 10:
                del_len = 1
            else:
                del_len = math.ceil(len_resp/10)
            action_percentile = percentileofscore([x for x in range(10)], action_index - 30)
            beg_index = int(action_percentile/100 * len_resp)

            if beg_index + del_len > len_resp:
                end_index = len_resp-1
            else:
                end_index = beg_index + del_len

            new_resp = resp_split[:beg_index] + resp_split[end_index:]

        if self.policy_type == 'PPO':
            self.state = self.create_state_cnn(' '.join(new_resp).strip('.'))
        elif self.policy_type == 'BERT':
            self.state, self.attention_mask = self.create_state_bert(' '.join(new_resp).strip('.'))
        self.response = ' '.join(new_resp).strip()

    def take_action(self, action_index):
        """
        0-16: delete indices action*3:action*3+3
        17-34: add phrase 1 after index
        35:52 add phrase 2 after index
        53:70 add phrase 3 after index
        71:88 add phrase 4 after index
        89:106 add phrase 5 after index
        """
        self.response = re.sub('[^0-9a-zA-Z]+', ' ', self.response)
        self.response = self.response.replace('.', '. ').replace(',', ' ').replace('-', ' ').replace('  ', ' ').strip()
        resp_split = [x.lower() for x in self.response.split(' ') if x not in ['', '.', ',']]
        len_resp = len(resp_split)

        if action_index < 17:
            # remove appropriate words
            beg_index, end_index = action_index * 3, action_index * 3 + 3
            if len_resp == 1:
                new_resp = resp_split
            elif len_resp > end_index:
                new_resp = resp_split[0:beg_index] + resp_split[end_index:]
            elif len_resp == 0:
                new_resp = 'the'
            else:
                new_resp = resp_split
            # set new state
            self.state = self.create_state_cnn(' '.join(new_resp).strip('.'))
        else:
            # get phrase to add
            if 35 > action_index > 16:
                phrase = self.phrases_list[0]
                phrase_index = action_index - (17 * 1) + 1
            elif 53 > action_index > 34:
                phrase = self.phrases_list[1]
                phrase_index = action_index - (17 * 2) + 1
            elif 71 > action_index > 52:
                phrase = self.phrases_list[2]
                phrase_index = action_index - (17 * 3) + 1
            elif 89 > action_index > 70:
                phrase = self.phrases_list[3]
                phrase_index = action_index - (17 * 4) + 1
            elif 107 > action_index > 88:
                phrase = self.phrases_list[4]
                phrase_index = action_index - (17 * 5) + 1
            # below is new
            elif 125 > action_index > 106:
                phrase = self.phrases_list[5]
                phrase_index = action_index - (17 * 6) + 1
            elif 143 > action_index > 124:
                phrase = self.phrases_list[6]
                phrase_index = action_index - (17 * 7) + 1
            elif 161 > action_index > 142:
                phrase = self.phrases_list[7]
                phrase_index = action_index - (17 * 8) + 1
            else:
                phrase = self.phrases_list[8]
                phrase_index = action_index - (17 * 9) + 1

            # add phrase
            if len_resp == 0:
                new_resp = phrase
            elif len_resp < phrase_index or len_resp > 47:
                new_resp = resp_split
            else:
                new_resp = resp_split[:phrase_index] + [phrase] + resp_split[phrase_index:]

            self.state = self.create_state_cnn(' '.join(new_resp).strip('.'))

        self.response = ' '.join(new_resp).strip()

    def observe_reward(self):
        """
        Calculate current loss, subtract from previous loss
        and update previous loss with current loss.
        Subtract one to penalize each timestep needed.
        """
        self.new_prediction = self.get_prediction(self.response)
        #if np.all(self.state == np.zeros(400)):  # this needs to be fixed for the cnn version
        #    reward = -1.3

        if np.all(self.state[0, :] == np.zeros(self.state.shape[1])):
            reward = -1.3  # used to be 1.3
            return reward

        ex_new = self.new_prediction[1] * 1 + self.new_prediction[2] * 2 + self.new_prediction[3] * 3 + self.new_prediction[4] * 4
        ex_old = self.old_prediction[1] * 1 + self.old_prediction[2] * 2 + self.old_prediction[3] * 3 + self.old_prediction[4] * 4

        if ex_new - ex_old == 0:  # case where the agent deletes empty part
            reward = -1.2
        elif ex_new > self.prob_threshold:  # terminal state
            reward = 1
        else:
            reward = .5 * (ex_new - ex_old) - 1  # multiplying by 1 instead of 3

        return reward

    def check_done(self):
        """ Check if kappa score has reached pre-defined threshold """
        ex_new = self.new_prediction[1] * 1 + self.new_prediction[2] * 2 + self.new_prediction[3] * 3 + self.new_prediction[4] * 4
        if ex_new > self.prob_threshold:
            #print(ex_new, self.new_prediction)
            done = 1
        else:
            done = 0
        return done

    def reset(self):
        """
        Returns the initial state & corresponding q_input (state and all possible actions)
        """
        self.init_state()
        return self.state, self.response, self.attention_mask

    def reset_test(self):
        """
        Returns the initial state & corresponding q_input (state and all possible actions)
        """
        self.init_state_test()
        return self.state, self.response, self.attention_mask

    def step(self, action_index, old_action):
        """
        returns next_state, reward, done, self.response
        action is just the index of action
        """
        self.action_index = action_index
        self.old_action = old_action
        self.take_action2(action_index=action_index)
        reward = self.observe_reward()
        self.old_prediction = self.new_prediction
        next_state = self.state
        done = self.check_done()
        return next_state, reward, done, self.response


class Agent():
    """Interacts with and learns form environment."""

    def __init__(self, embed_dim, seed, num_classes):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            seed (int): random seed
        """

        self.embed_dim = embed_dim
        self.seed = random.seed(seed)
        self.num_classes = num_classes

        #self.qnetwork_local = QNetwork(state_size, seed).to(device)
        #self.qnetwork_target = QNetwork(state_size, seed).to(device)

        self.qnetwork_local = QNET_CNN(embed_dim=embed_dim, num_classes=num_classes).to(device)  # need to add seed option..
        self.qnetwork_target = QNET_CNN(embed_dim=embed_dim, num_classes=num_classes).to(device)
        self.qnetwork_critic = QNET_CNN(embed_dim=embed_dim, num_classes=1)

        self.bert_actor = BERTPolicy(num_classes=num_classes).to(device)
        self.bert_critic = BERTPolicy(num_classes=1)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action_representation, reward, next_state, done):
        self.memory.add(state, action_representation, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def act(self, state, eps=0, return_logprobs=False):
        """
        Returns action for given state as per current policy

        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state)
        state = state.to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.float()(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            action_index = np.argmax(action_values.cpu().data.numpy())
            if return_logprobs:
                return action_index, action_values[action_index]
            else:
                return action_index

        else:
            action_index = random.choice(np.arange(len(action_values)))
            if return_logprobs:
                return action_index, action_values[action_index]
            else:
               return action_index

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        #criterion = torch.nn.HuberLoss()
        criterion = torch.nn.MSELoss() # should this be cross entropy loss?
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        predicted_targets = self.qnetwork_local(states)
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_states).detach()
        labels = rewards + (gamma * labels_next * (1 - dones))
        loss = criterion(predicted_targets, labels.to(device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  # update target

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
