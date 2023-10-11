# from tensorflow.keras import Input
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Add
# from tensorflow.keras.initializers import VarianceScaling
# from tensorflow.keras.backend import mean
# from tensorflow.keras.losses import Huber
# from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam
import pdb
# import tensorflow as tf

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 2, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, bias=False)

        self.linear1 = nn.Linear(16384, 256)
        self.linear2 = nn.Linear(16384, 256)

        self.final_linear1 = nn.Linear(256, 4)
        self.final_linear2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)

        x = nn.Flatten()(x)

        x1 = nn.ReLU()(self.linear1(x))
        x2 = nn.ReLU()(self.linear2(x))

        advantage = self.final_linear1(x1)
        value = self.final_linear2(x2)

        return advantage - value
    
class DQN:
    def __init__(self, learning_rate = 0.00001, action_space = 4, gamma = 0.99, batch_size = 32, device = 'cuda'):
        self.lr = learning_rate
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = self.create_model().to(device)
        self.target_model = self.create_model().to(device)
        self.model.train()
        self.target_model.eval()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.device = device
        self.update()

    def load(self, name):
        print(self.device)
        self.model.load_state_dict(torch.load(name, map_location = torch.device(self.device)))
        self.update()

    def create_model(self):
        return CNN()

    def predict(self, state):
        results = self.model(torch.tensor(state.astype(np.float32), device=self.device))
        return results
    
    def learn(self, transitions, verbose = 0):
        states, actions, rewards, new_states, terminals = transitions
        # pdb.set_trace()
        states = torch.tensor((states / 255.0).astype(np.float32), device=self.device)
        new_states = torch.tensor((new_states / 255.0).astype(np.float32), device=self.device)

        self.optimizer.zero_grad()
        future_qs = (self.target_model(new_states))
        expected_qs = (self.model(states))
        new_qs = expected_qs.clone()
        for i in range(len(expected_qs)):
            if terminals[i]:
                new_qs[i][actions[i]] = rewards[i]
            else:
                new_qs[i][actions[i]] = rewards[i] + self.gamma * max(future_qs[i])
        loss = self.loss_fn(expected_qs, new_qs)
        loss.backward()
        if verbose:
            print(loss.item())
        self.optimizer.step()
        

    # def learn_old(self, transitions, verbose = 0):
    #     states, actions, rewards, new_states, terminals = transitions
    #     future_qs = (self.target_model(new_states / 255.0)).numpy()

    #     expected_qs = (self.model(states / 255.0)).numpy()
    #     for i in range(len(expected_qs)):
    #         if terminals[i]:
    #             expected_qs[i][actions[i]] = rewards[i]
    #         else:
    #             expected_qs[i][actions[i]] = rewards[i] + self.gamma * max(future_qs[i])
    #     self.model.fit(states / 255.0, expected_qs, verbose=verbose)

    def update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def test_model():
    from game import Game
    import numpy as np
    env = Game()
    state = env.reset()
    dqn = DQN()
    print(dqn.model(np.array([[state for _ in range(4)]]).reshape((1, 4, 84, 84))/255.0))


# def test_train():
#     from game import Game
#     from RM import ReplayMemory
#     import time
#     g = Game()
#     rm = ReplayMemory()
#     for i in range(32):
#         done = False
#         _, frame = g.reset()
#         while not done:
#             pdb.set_trace()
#             _, reward, done, new_frame = g.step(1)
#             rm.add_experience(1, frame, reward, done)
#             frame = new_frame
#     dqn = DQN()
#     for _ in range(100):
#         last = time.time()
#         transitions = rm.get_minibatch()
#         dqn.learn(transitions)
#         print('Time cost: {}'.format(time.time() - last))
#     transitions = rm.get_minibatch()

#     dqn.update()
#     dqn.save('simple_test')
#     print(dqn.model(transitions[0][0].reshape(1, 4, 84, 84) / 255.0))
#     dqn.model.load_state_dict(torch.load('simple_test'))
#     print(dqn.model(transitions[0][0].reshape(1, 4, 84, 84) / 255.0))

#     for _ in range(100):
#         last = time.time()
#         transitions = rm.get_minibatch()
#         dqn.learn(transitions)
#         print('Time cost: {}'.format(time.time() - last))
#     dqn.update()

# test_train()

