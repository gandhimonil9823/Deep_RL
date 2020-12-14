# @article{wang2018deep,
#   title={Deep Reinforcement Learning of Cell Movement in the Early Stage of C. elegans Embryogenesis},
#   author={Wang, Zi and Wang, Dali and Li, Chengcheng and Xu, Yichi and Li, Husheng and Bao, Zhirong},
#   journal={arXiv preprint arXiv:1801.04600},
#   year={2018}
# }

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

index_value = 0
number_of_states = []
number_of_actions = []

# Initialize the neural network class for Pytouch
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        # three hidden layers, with 512, 1024 and 1024 nodes
        self.fc1 = torch.nn.Linear(number_of_states, 512)
        self.fc2 = torch.nn .Linear(512, 1024)
        self.fc3 = torch.nn .Linear(1024, 1024)
        self.out = torch.nn .Linear(1024, number_of_actions)

# Forward propagate the state of the agent at a time t
    # Uing ReLu as an activatio function
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.out(x)
        return value


# Initilizing the Deep Q-Network network
class DQN(object):
    def __init__(self):
        # creating two neural networks according to the target netwrok optimization for the Deep Q-Network network
        self.prediction_network, self.target_network = Net(), Net()

        # epsilon greedy strategy, which not only helps us in terms of the previous experiences for rewards maximization, but also helps choosing the states randomdly
        self.epsilon_greedy_value = 0.3

        # Learning rate for the neural network
        self.learning_rate_value = 0.01
        self.optimizer = torch.optim.Adam(self.prediction_network.parameters(), lr=self.learning_rate_value)

        # Error measure
        self.mean_square_loss_func = nn.MSELoss()

        # Experience replay - it is kind of a memory where we store the transitions of the agents
        self.memory = np.zeros((number_of_states * 2))

# returns the max Q value from the output of the neural network
    def get_max_of_action(self, input_array):
        if np.random.uniform() < self.epsilon_greedy_value:
            value = self.prediction_network.forward(input_array)
            action_to_take = torch.max(value, 1)[1].data.numpy()[0]
            return action_to_take

# save the training samples which are the St, At, Rt and St + 1
    def save_transition(self, state, action, reward, start_next):
        transition = np.hstack((state, [action, reward], start_next))
        self.memory[index_value, :] = transition

