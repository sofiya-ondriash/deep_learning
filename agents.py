import torch
import numpy as np
from torch import nn

from models import DenseModel, CNNModel

class Memory(object):
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = list()

    def remember(self, m):
        if len(self.memory)==self.max_memory:
            del self.memory[0]
        self.memory.append(m)

    def random_access(self, batch_size=None):
        if batch_size is not None:
            indices = np.random.randint(0,len(self.memory),batch_size)
            s_, n_s_, a_, r_, game_over_ = zip(*[self.memory[i] for i in indices])
            s_ = torch.stack(s_)
            n_s_ = torch.stack(n_s_)
            a_ = torch.vstack(a_)
            r_ = torch.tensor(r_, device=a_.device).unsqueeze(1)
            game_over_ = torch.tensor(game_over_, device=a_.device).unsqueeze(1)
            return s_, n_s_, a_, r_, game_over_
        else: 
            return self.memory[np.random.randint(0,len(self.memory))]
    

class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.epsilon = args.epsilon
        self.n_action = 4

    def forward(self, s):
        return self.act(s)

    def set_epsilon(self,e):
        self.epsilon = e

    def act(self,s):
        """ This function should return the next action to do:
        an integer between 0 and 4 (not included) with a random exploration of epsilon"""
        if self.training:
            if np.random.rand() <= self.epsilon:
                a = torch.randint(0, self.n_action, (1,), device=s.device)
            else:
                a = self.learned_act(s)
        else: 
            a = self.learned_act(s)
        return a

    def learned_act(self,s):
        """ Act via the policy of the agent, from a given state s
        it proposes an action a"""
        pass

    def reinforce(self, s, n_s, a, r, game_over_):
        """ This function is the core of the learning algorithm. 
        It takes as an input the current state s_, the next state n_s_
        the action a_ used to move from s_ to n_s_ and the reward r_.
        
        Its goal is to learn a policy.
        """
        pass


class StraightAgent(Agent):
    def __init__(self, args,**kwargs):
        super(StraightAgent, self).__init__( args,**kwargs)  
        self.trainable = False

    def learned_act(self, s):
        return torch.tensor(0)


class RandomAgent(Agent):
    def __init__(self, args,**kwargs):
        super(RandomAgent, self).__init__( args,**kwargs)  
        self.trainable = False

    def learned_act(self, s):
        ### To do 10
        return torch.randint(0,self.n_action,(1,))


class DQN(Agent):
    def __init__(self, args):
        super(DQN, self).__init__(args)

        # Discount for Q learning
        self.discount = 0.99

        # Number of state
        self.n_state = 2 + args.explore*1
        self.grid_size = args.grid_size
        
        # Memory
        self.memory = Memory(args.memory_size)
        
        # Batch size when learning
        self.batch_size = args.batch_size

        # Load model
        if args.agent == 'fc':
            self.model = DenseModel(self.n_state, self.n_action)
        elif args.agent == 'cnn':
            self.model = CNNModel(self.n_state, self.n_action)
        
        self.trainable = True
        
        
    def learned_act(self, s):
        raise NotImplementedError
        ### Todo 15

    def reinforce(self, s_, n_s_, a_, r_, game_over_):
        # Two steps: first memorize the states, second learn from the pool, batched
        self.memory.remember([s_, n_s_, a_, r_, game_over_])
        
        replays = self.memory.random_access(self.batch_size)
        s_, n_s_, a_, r_, game_over_ = replays
        
        # Initialize input states and target_q with correct shapes
        input_states = s_.clone()
        target_q = self.model(input_states).clone()

        # Calculate target values where game is not over
        not_done = ~game_over_
        target = r_.clone()
        target[not_done] += self.discount * torch.max(self.model(n_s_), dim=1)[0].unsqueeze(1)[not_done]

        # Update target_q for the actions taken
        mask_a = torch.zeros(target_q.shape, device=target_q.device)
        mask_a[range(mask_a.shape[0]), a_.view(-1)] = 1
        target_q = (1 - mask_a) * target_q + mask_a * target
        
        # HINT: Clip the target to avoid exploiding gradients.. -- clipping is a bit tighter
        target_q = torch.clamp(target_q, -3, 3)
        raise NotImplementedError
        ### Todo 16



