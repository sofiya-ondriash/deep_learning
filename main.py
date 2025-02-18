import os
import torch
import argparse

from environment import Environment
from train_utils import train_model, test_model
from agents import RandomAgent, StraightAgent, DQN
from utils import get_optimizer, save_model, plot_model, logger

def main(args): 

    # Create the folder where to store the results
    os.makedirs(os.path.join("experiments", args.path), exist_ok=True)

    # Initialize the logger
    log = logger(args)
    log("Starting the training of the agent") # type: ignore

    # Set the device (cuda, cpu or mps)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log("Device used: {}".format(device)) # type: ignore

    # Initialize the game
    env = Environment(args=args, device=device)

    # Initialize the agent
    if args.agent == 'random':
        agent = RandomAgent(args)
    elif args.agent == 'straight':
        agent = StraightAgent(args)
    elif args.agent == 'fc' or args.agent == 'cnn':
        agent = DQN(args).to(device)

    # Train the agent if needed
    if agent.trainable:
        optimizer = get_optimizer(model = agent.model, lr=args.lr, optimizer=args.optimizer)
        train_model(env=env, agent=agent, optimizer=optimizer, args=args, log=log)

    # Test the agent
    log("Eval model:")
    test_model(env=env, agent=agent, args=args, log=log)
    plot_model(env=env, args=args, name='final')
    if agent.trainable:
        save_model(agent=agent, args=args, name='final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Deep Reinforcement Learning for the game of DQN.
                    Example of usage:
                    python main.py --path base --agent fc --memory_size 1000 --explore --lr 0.1 --epoch 300 --optimizer SGD --epoch_eval 10 --freq 50""")

    # General parameters
    parser.add_argument('--path', type=str, default='base', help='Path to the folder where to model and plots')
    parser.add_argument('--log_type', type=str, default='print', choices=['print', 'file'], help='Type of log to use')

    # Environment parameters
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature when generating the map')
    parser.add_argument('--grid_size', type=int, default=13, help='Size of the grid') 
    
    ##TO DO 3
    
    parser.add_argument('max_time', type = int, default=100, help='Max_time')


    # Agent parameters
    parser.add_argument('--agent', type=str, default='straight', choices=['straight', 'random', 'fc', 'cnn'], help='Type of agent to use')
    parser.add_argument('--memory_size', type=int, default=1000, help='Size of the memory buffer')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Exploration parameter') 
    parser.add_argument('--explore', default=False, action='store_true', help='Whether to use an epsilon-greedy strategy')

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'], help='Type of optimizer to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of the batch')

    # Evaluation parameters
    parser.add_argument('--epoch_eval', type=int, default=10, help='Number of epochs for evaluation')
    parser.add_argument('--freq', type=int, default=20, help='Frequency of evaluation/plot')
    
    args = parser.parse_args()



    main(args)
