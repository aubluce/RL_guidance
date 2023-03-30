
from rl_functions import Environment, Agent
from bert_prediction import BertClassificationPredictor
from utils import process_action_space, process_test_data, train_w2v_model
from ppo import PPO

from transformers import BertTokenizer
import sys

#  all other imports
import gensim.downloader
import pandas as pd
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#  define some necessary paths and file names
data_path = 'data/LinnData/' + str(sys.argv[1]) + '/'
train_name = 'train.csv'
val_name = 'val.csv'
test_name = 'test.csv'
action_space = 'action2.csv'
autograding_model_path = 'output/ASAG_models/MIFreq_KI_5cat_031823/pytorch_model.bin'

#  read in data for training and action space key phrases
test_df = pd.read_csv(data_path + test_name)
train_df = pd.read_csv(data_path + train_name)
val_df = pd.read_csv(data_path + val_name)

combined_df = pd.concat([test_df, val_df, train_df])
#processed_df = process_test_data(test_df)
processed_combined = process_test_data(combined_df)
processed_df = process_test_data(train_df)
processed_test = process_test_data(pd.concat([val_df, test_df]))
action_df = pd.read_csv(data_path + action_space)
key_phrase_list = process_action_space(action_df)

#  w2v and glove models for state space representation of student responses
w2v_model = train_w2v_model(processed_combined, key_phrase_list, vector_size=10, model='fasttext')
glove_model = gensim.downloader.load('glove-wiki-gigaword-50')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


#  load the pre-trained auto grader
predictor = BertClassificationPredictor(
    #model_path=autograding_model_path + 'pytorch_model.bin',
    model_path=autograding_model_path,
    label_path=data_path,
    multi_label=True,
    model_type='bert',
    do_lower_case=False,)


prob_threshold = 3.3
old_action = None
policy_type = 'BERT'
env = Environment(processed_combined, processed_test, prob_threshold, predictor, w2v_model,
                  glove_model, key_phrase_list, key_phrase_list, old_action, bert_tokenizer, policy_type)
# num_classes is 119 if individual words. 40 if phrases.
agent = Agent(embed_dim=11, seed=sys.argv[2], num_classes=40)

#dqn(env, agent, n_episodes=100000, max_t=12, eps_start=1.0, eps_end=0.05, eps_decay=0.9999,
#    output_path='output/rl_scores/' + sys.argv[1] + '/', plot_rolling_n=300, train_episodes=75000)

# Algorithm hyperparameters
timesteps_per_batch = 84  # Number of timesteps to run per batch
max_timesteps_per_episode = 8  # Max number of timesteps per episode
total_timesteps = 300000
updates_per_iteration = 4  # Number of times to update actor/critic per iteration # was 5
lr = 0.001  # Learning rate of actor optimizer
gamma = 0.99  # Discount factor to be applied when calculating Rewards-To-Go
clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
# Miscellaneous parameters
render = True  # If we should render during rollout
render_every = 10  # Only render every n iterations
save_freq = 1000  # How often we save in number of iterations
seed = 15


ppo = PPO(env, agent, updates_per_iteration, max_timesteps_per_episode, total_timesteps, timesteps_per_batch, lr, gamma, clip, render,
                    render_every, save_freq, seed, policy_type)
# do ittttttttttttt
ppo.ppo()
