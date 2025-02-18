from utils import save_model, plot_model

def train_model(env, agent, optimizer, args, log):   
    for e in range(args.epoch):
        if args.explore:
            raise NotImplementedError
            ### Todo 21

        # At each epoch, we restart to a fresh game and get the initial state
        state = env.reset()
        # This assumes that the games will end
        game_over = False

        win = 0
        lose = 0

        while not game_over:
            if args.explore:
                raise NotImplementedError
                ### Todo 21

            optimizer.zero_grad()

            # The agent performs an action
            action = agent(state)
            # Apply an action to the environment, get the next state, the reward
            # and if the games end
            prev_state = state
            state, reward, game_over = env.act(action)

            # Update the counters
            if reward >= 0:
                win = win + reward
            if reward < 0:
                lose = lose -reward

            # Apply the reinforcement strategy
            loss = agent.reinforce(prev_state, state,  action, reward, game_over)
            loss.backward()
            optimizer.step()

        log("Epoch {:03d}/{:03d} | Loss {:.4f} | Win/lose count {:.1f}/{:.1f} ({:.1f})"
                .format(e, args.epoch, loss, win, lose, win-lose))

        # Save as a mp4
        if e % args.freq == 0 and e < args.epoch-1:
            test_model(env=env, agent=agent, args=args, log=log)
            plot_model(env=env, args=args, name='epoch_{}'.format(e))
            save_model(agent=agent, args=args, name='epoch_{}'.format(e))


def test_model(env, agent, args, log):
    # Number of won games
    agent.eval()
    env.eval()
    score = 0
     
    for e in range(args.epoch_eval):
        win = 0
        lose = 0
        state = env.reset()
        state, reward, game_over = env.act(agent(state))
        while (game_over == 0):
            state, reward, game_over = env.act(agent(state))
            if reward > 0.1:
                win += reward
            else :
                if reward < -0.1:
                    lose -= reward
        
        # Update stats
        score = score + win-lose
    log('\tEval score: '+str(score/args.epoch_eval))
    agent.train()
    env.train()
