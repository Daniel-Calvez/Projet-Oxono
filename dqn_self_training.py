'''
Train loop for the Deep-Q-Learning network against itself
M1 Bio-info project
Daniel Calvez & Vincent Ducot
2025
Using pytorch
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
from collections import deque
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim
import ia
import model
import dqn

class ReplayMemory:
    '''
    Class to keep memory of games
    '''
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        '''
        Add a move in the memory
        '''
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''
        Randomly pick some moves in the memory
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(network, board, player1, player2, current_player, epsilon, device):
    '''
    Choose an action with epsilon-greedy
    
    Args:
        The DQN network as an XonoxNetwork object
        The board as a matrix
        Player1's name
        Player2's name
        Current player's name
        Probability of random action 
        Device use to compute (CPU or GPU)
        
    Returns:
        A tuple with the choosen action and its index
    '''
    state_tensor = dqn.convert_board(board, player1, player2, current_player)
    state = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)

    # Epsilon-greedy
    if random.random() < epsilon:
        # Random move
        action = ia.random_play(board, 0, current_player)
        action_idx = -1
        if action:
            for i in range(2592):
                if dqn.traduce_output(i) == action:
                    action_idx = i
                    break
    else:
        # Action based on the network output
        with torch.no_grad():
            q_values = network(state).squeeze(0)
        action_table = dqn.filter_outputs(q_values, board, current_player)
        if not action_table[0]:  # Should not happen
            action = ia.random_play(board, 0, current_player)
            action_idx = -1
            if action:
                for i in range(2592):
                    if dqn.traduce_output(i) == action:
                        action_idx = i
                        break
        else:
            # Choose the action with the best score
            best_idx = torch.argmax(torch.tensor(action_table[1])).item()
            action = action_table[0][best_idx]
            action_idx = -1
            for i in range(2592):
                if dqn.traduce_output(i) == action:
                    action_idx = i
                    break

    return action, action_idx, state

def train_selfplay(num_epochs=100, batch_size=16, gamma=0.99, epsilon_start=1.0,
             epsilon_end=0.1, epsilon_decay=0.995, learning_rate=0.001,
             save_path='xonox.dqn', save_interval=20, eval_interval=25):
    '''
    Train the DQN network to play against random player
    
    Args:
        num_epochs: Number of epochs
        batch_size: Size of batchs for the training
        gamma: Reduction factor for rewards
        epsilon_start: Initial probability of random actions 
        epsilon_end: Minimal probability of random actions 
        epsilon_decay: Epsilon decline rate
        learning_rate: Learning rate
        save_path: Filepath of the model
        save_interval: Number of epochs between saving the model 
        eval_interval: Number of epochs between evaluating the model
    Returns
        The trained network
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init the network
    network = dqn.load_cnn(save_path) if Path(save_path).exists() else dqn.XonoxNetwork()
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    memory = ReplayMemory()

    epsilon = epsilon_start
    total_steps = 0
    p1_wins = 0
    p2_wins = 0
    draws = 0

    for epoch in range(num_epochs):
        # Init a new game
        board = model.init_board()
        player1 = "DQN_1"
        player2 = "DQN_2"
        current_player = player1 if random.random() < 0.5 else player2
        done = False

        epoch_experiences = []

        print(f"Epoch {epoch+1}/{num_epochs}, Epsilon: {epsilon:.4f}")

        # Each turn in the game
        while not done:
            # Select an action for the current player
            action, action_idx, state = select_action(
                network, board, player1, player2, current_player, epsilon, device
            )

            # Execute the move
            totem_coord = model.convert_coord(action[1:3])
            totem = 'T_' + action[0]
            model.move_totem(board, totem, totem_coord)
            coord_action = model.convert_coord(action[3:])
            board[coord_action[0]][coord_action[1]] = current_player[0] + '_' + action[0]

            # Check if the game is over
            nb_free_cells = len(model.all_free_cells(board))
            if model.is_winner(board, current_player, coord_action):
                done = True
                if current_player == player1:
                    p1_wins += 1
                    # Reward for player 1, penality for player 2
                    moves_nb = (34 - nb_free_cells) // 2
                    reward_p1 = 1.0 - moves_nb / 16 * 0.2
                    reward_p2 = -1.0 + moves_nb / 16 * 0.2
                else:
                    p2_wins += 1
                    # Reward for player 1, penality for player 2
                    moves_nb = (34 - nb_free_cells) // 2
                    reward_p2 = 1.0 - moves_nb / 16 * 0.2
                    reward_p1 = -1.0 + moves_nb / 16 * 0.2
            elif nb_free_cells == 2:
                done = True
                draws += 1
                reward_p1 = 0.0
                reward_p2 = 0.0
            else:
                # Game still ongoing, encourage fast games
                reward_p1 = -0.01
                reward_p2 = -0.01

            next_player = player2 if current_player == player1 else player1

            next_state_tensor = dqn.convert_board(board, player1, player2, next_player)
            next_state = torch.from_numpy(next_state_tensor).float().unsqueeze(0).to(device)


            # Store turn data in memory
            if action_idx != -1:
                reward = reward_p1 if current_player == player1 else reward_p2
                epoch_experiences.append((state.cpu(), action_idx, reward, next_state.cpu(), done, current_player))

            # Change player
            current_player = next_player

            total_steps += 1

        for exp in epoch_experiences:
            memory.push(exp[0], exp[1], exp[2], exp[3], exp[4])

        # Train the network if there are enough games for a batch
        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones, _ = zip(*batch)

            states = torch.cat(states).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.cat(next_states).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)

            # Compute current Q values
            q_values = network(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q values
            with torch.no_grad():
                next_q_values = network(next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q_values = rewards + gamma * max_next_q * (~dones)

            # Compute loss
            loss = F.smooth_l1_loss(q_values, target_q_values)

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Save the model to the file
        if (epoch + 1) % save_interval == 0:
            dqn.write_cnn(network, save_path)
            print(f"Model saved at epoch {epoch+1}")

        # Model evaluation
        if (epoch + 1) % eval_interval == 0:
            p1_win_rate = p1_wins / (p1_wins + p2_wins + draws) if (p1_wins + p2_wins + draws) > 0 else 0
            print(f"Epoch {epoch+1}: P1 wins: {p1_wins}, P2 wins: {p2_wins}, Draws: {draws}, Win rate: {p1_win_rate:.4f}")
            p1_wins, p2_wins, draws = 0, 0, 0

    # Final save of the model
    dqn.write_cnn(network, save_path)
    print(f"Training over, save the model to {save_path}")
    return network

def evaluate_against_random(model_path, num_games=100):
    '''
    Evaluate the DQN model's performance against a random player
    
    Args:
        Filepath of the model
        Number of games to play for the evaluation
    
    Returns:
        Winning rate of the DQN model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = dqn.load_cnn(model_path)
    network.to(device)
    network.eval()

    wins = 0
    losses = 0
    draws = 0

    for game in range(num_games):
        board = model.init_board()
        player1 = "DQN"
        player2 = "Random"
        current_player = player1 if game % 2 == 0 else player2
        done = False

        while not done:
            if current_player == player1:  # DQN turn
                state_tensor = dqn.convert_board(board, player1, player2, current_player)
                state = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    q_values = network(state).squeeze(0)

                action_table = dqn.filter_outputs(q_values, board, current_player)
                if not action_table[0]:
                    draws += 1
                    break

                best_idx = torch.argmax(torch.tensor(action_table[1])).item()
                action = action_table[0][best_idx]

                # Execute action
                totem_coord = model.convert_coord(action[1:3])
                totem = 'T_' + action[0]
                model.move_totem(board, totem, totem_coord)
                coord_action = model.convert_coord(action[3:])
                board[coord_action[0]][coord_action[1]] = current_player[0] + '_' + action[0]

                nb_free_cells = len(model.all_free_cells(board))
                if model.is_winner(board, current_player, coord_action):
                    done = True
                    wins += 1

                elif nb_free_cells == 2:
                    done = True
                    draws += 1

                current_player = player2

            else:  # Random player turn
                action = ia.random_play(board, 0, current_player)
                if not action:
                    draws += 1
                    break

                # Execute action
                totem_coord = model.convert_coord(action[1:3])
                totem = 'T_' + action[0]
                model.move_totem(board, totem, totem_coord)
                coord_action = model.convert_coord(action[3:])
                board[coord_action[0]][coord_action[1]] = current_player[0] + '_' + action[0]
                
                nb_free_cells = len(model.all_free_cells(board))
                if model.is_winner(board, current_player, coord_action):
                    done = True
                    losses += 1

                elif nb_free_cells == 2:
                    done = True
                    draws += 1

                current_player = player1

    win_rate = wins / num_games
    print(f"Evaluation on {num_games} games:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Winning rate: {win_rate:.4f}")

    return win_rate

if __name__ == "__main__":
    MODEL_PATH = "oxono_selfplay.pth"

    # Model's training by self-play
    trained_network = train_selfplay(
        num_epochs=500,
        batch_size=64,
        gamma=0.99,
        epsilon_start=0.5,  # Less exploration, as the model is already trained
        epsilon_end=0.05,
        epsilon_decay=0.995,
        learning_rate=0.0005,  # Lower rate to improve the model
        save_path=MODEL_PATH
    )

    # Model's evaluation
    evaluate_against_random(MODEL_PATH, num_games=100)
