import torch
from torch import nn
import scipy
import numpy as np

class Environment(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Environment, self).__init__()
        grid_size = args.grid_size+4
        self.grid_size = grid_size
        # Check if args.max_time is defined
        if not hasattr(args, 'max_time'):
            raise ValueError("args.max_time is not defined")
        self.max_time = args.max_time
        self.temperature = args.temperature
        self.device = device
        self.args = args

        # board on which one plays
        self.board = torch.zeros((grid_size,grid_size), device=device)
        self.position = torch.zeros((grid_size,grid_size), device=device)

        self.x = 0
        self.y = 0

        # self time
        self.t = 0

        self.scale = 16
        self.frames = []
        self.to_draw = np.zeros((args.max_time+2, grid_size*self.scale, grid_size*self.scale, 3))

    def get_state(self):
        state = torch.cat((self.board.reshape(self.grid_size, self.grid_size,1),
                        self.position.reshape(self.grid_size, self.grid_size,1)), dim=2)
        if self.args.explore:
            state = torch.cat((self.board.reshape(self.grid_size, self.grid_size,1),
                            self.position.reshape(self.grid_size, self.grid_size,1),
                            self.malus_position.reshape(self.grid_size, self.grid_size,1)),dim=2)
        state = state[self.x - 2:self.x + 3, self.y - 2:self.y + 3, :]
        return state

    def get_frame(self,t):
        b = torch.zeros((self.grid_size,self.grid_size,3))

        # Set the position of the cheese
        mask_cheese = self.board.cpu() > 0
        b += torch.stack([244*mask_cheese, 67*mask_cheese, 54*mask_cheese], dim=2)

        # Set the position of the poison
        mask_poison = self.board.cpu() < 0
        b += torch.stack([8*mask_poison, 69*mask_poison, 126*mask_poison], dim=2)

        # Set the position of the mouse
        mask_mouse = torch.zeros((self.grid_size, self.grid_size))
        mask_mouse[self.x, self.y] = 1
        b += 255*mask_mouse.unsqueeze(2)

        # Fill the rest of the board by 128
        b += 128*(1-(mask_poison+mask_cheese+mask_mouse).unsqueeze(2))

        # Set the border to black
        b[-2:,:,:] = 0
        b[:,-2:,:] = 0
        b[:2,:,:] = 0
        b[:,:2,:] = 0

        # Rescale the image
        b = scipy.ndimage.zoom(b.numpy(), (self.scale, self.scale, 1), order=0)  
        self.to_draw[t,:,:,:] = b

    def act(self, action):
        """This function returns the new state, reward and decides if the
        game ends."""

        self.get_frame(int(self.t))

        self.position = torch.zeros((self.grid_size, self.grid_size), device=self.device)

        self.position[0:2,:]= -1
        self.position[:,0:2] = -1
        self.position[-2:, :] = -1
        self.position[-2:, :] = -1

        if action == 0:
            if self.x == self.grid_size-3:
                self.x = self.x-1
            else:
                self.x = self.x + 1
        elif action == 1:
            if self.x == 2:
                self.x = self.x+1
            else:
                self.x = self.x-1
        elif action == 2:
            if self.y == self.grid_size - 3:
                self.y = self.y - 1
            else:
                self.y = self.y + 1
        elif action == 3:
            if self.y == 2:
                self.y = self.y + 1
            else:
                self.y = self.y - 1
        else:
            RuntimeError('Error: action not recognized')

        self.t = self.t + 1
        reward = self.board[self.x, self.y].item() 

        if self.args.explore and self.training:
            raise NotImplementedError
            ### Todo 22

        self.board[self.x, self.y] = 0
        self.position[self.x, self.y] = 1 # Update the visited position state
        self.malus_position[self.x, self.y] += 1

        game_over = self.t > self.max_time

        ### Todo 13
        state = torch.zeros(1)    

        return state, reward, game_over

    def reset(self):
        """This function resets the game and returns the initial state"""

        # Reset the last game
        self.to_draw = np.zeros((self.max_time+2, self.grid_size*self.scale, self.grid_size*self.scale, 3))
        self.t = 0

        # Randomly generate the position of the mouse, the cheese and the poison
        self.x = np.random.randint(3, self.grid_size-3, size=1)[0]
        self.y = np.random.randint(3, self.grid_size-3, size=1)[0]

        bonus = torch.bernoulli(self.temperature*torch.ones((self.grid_size, self.grid_size), device=self.device))
        malus = -torch.bernoulli(self.temperature*torch.ones((self.grid_size, self.grid_size), device=self.device))
        malus[bonus>0]=0 # No malus where there is a bonus

        self.malus_position = torch.zeros((self.grid_size, self.grid_size), device=self.device) 

        self.board = bonus + malus
        self.board[self.x,self.y] = 0 # No reward at the position of the mouse

        # Set the available positions and reset the visited positions
        self.position = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        self.position[0:2,:]= -1
        self.position[:,0:2] = -1
        self.position[-2:, :] = -1
        self.position[-2:, :] = -1

        ### Todo 13
        state = torch.zeros(1)

        return state
