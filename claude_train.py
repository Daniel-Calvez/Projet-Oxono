'''
Boucle d'entraînement pour le réseau Deep-Q-Learning d'Oxono
'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import model
from dqn import XonoxNetwork, convert_board, filter_outputs, traduce_output, load_CNN, write_CNN, print_tensor
from pathlib import Path
import ia
from matplotlib import pyplot as plt

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def train_dqn(num_episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, 
             epsilon_end=0.1, epsilon_decay=0.995, learning_rate=0.001, 
             save_path='oxono_dqn.pth', save_interval=100, eval_interval=50, eval_number = 50):
    """
    Entraîne le réseau DQN pour jouer à Oxono
    
    Args:
        num_episodes: Nombre d'épisodes d'entraînement
        batch_size: Taille des lots pour l'entraînement
        gamma: Facteur de réduction pour les récompenses futures
        epsilon_start: Probabilité initiale d'actions aléatoires
        epsilon_end: Probabilité minimale d'actions aléatoires
        epsilon_decay: Taux de diminution d'epsilon
        learning_rate: Taux d'apprentissage pour l'optimiseur
        save_path: Chemin pour sauvegarder le modèle
        save_interval: Intervalle d'épisodes entre les sauvegardes
        eval_interval: Intervalle d'épisodes entre les évaluations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialisation du réseau et de l'optimiseur
    network = load_CNN(save_path) if Path(save_path).exists() else XonoxNetwork()
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    
    # Mémoire d'expérience
    memory = ReplayMemory()
    
    # Variables de suivi
    epsilon = epsilon_start
    total_steps = 0
    wins = 0
    losses = 0
    draws = 0
    winrate= []
    for episode in range(num_episodes):
        # Initialisation d'une nouvelle partie
        board = model.init_board()
        player1 = "DQN"
        player2 = "Random"
        current_player = player1 if random.random() < 0.5 else player2
        done = False
        
        print(f"Épisode {episode+1}/{num_episodes}, Epsilon: {epsilon:.4f}")
        
        # Chaque tour
        while not done:
            if current_player == player1:  # Tour du DQN
                # Convertir l'état du jeu en tenseur
                state_tensor = convert_board(board, player1, player2, current_player)
                state = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
                # state = state[:, :, :, :6]  # On exclut la couche du joueur actif pour l'entrée du réseau
                # state = state.permute(0, 3, 1, 2)  # Format attendu par le CNN: [batch, channels, height, width]
                
                # Epsilon-greedy pour le choix de l'action
                if random.random() < epsilon:
                    # Action aléatoire
                    action = ia.random_play(board, 0, current_player)
                    # action = random.choice(valid_actions) if valid_actions else None
                else:
                    # Action basée sur le réseau
                    with torch.no_grad():
                        q_values = network(state).squeeze(0)
                    action_table = filter_outputs(q_values, board, current_player)
                    if not action_table[0]:  # Si aucune action valide n'est disponible
                        action = ia.random_play(board, 0, current_player)
                    else:
                        # Prendre l'action avec la plus grande valeur Q
                        best_idx = torch.argmax(torch.tensor(action_table[1])).item()
                        action = action_table[0][best_idx]
                
                # Exécuter l'action et obtenir le nouvel état
                #new_board = model.apply_action(board, action, current_player)
                totem_coord = model.convert_coord(action[1:3])
                totem = 'T_'+action[0]
                model.move_totem(board, totem, totem_coord)
                coord_action = model.convert_coord(action[3:])
                board[coord_action[0]][coord_action[1]] = current_player[0] + '_' + action[0]

                reward = 0

                # Vérifier si la partie est finie
                nb_free_cells = len(model.all_free_cells(board))
                if model.is_winner(board, current_player, coord_action):
                    done = True
                    moves_nb = (34 - nb_free_cells) // 2
                    reward = 1.0 - moves_nb / 16 * 0.2
                    wins += 1                        
                    # else:
                    #     moves_nb = nb_free_cells // 2
                    #     reward = -1.0 + moves_nb / 16 * 0.2
                    #     losses += 1
                        
                elif nb_free_cells == 2:
                    # print("Aucune action valide disponible, partie nulle.")
                    reward = 0
                    done = True
                    draws += 1
                    continue
                
                # Convertir le nouvel état en tenseur
                next_state_tensor = convert_board(board, player1, player2, player2)  # Prochain joueur
                next_state = torch.from_numpy(next_state_tensor).float().unsqueeze(0).to(device)
                # next_state = next_state[:, :, :, :6]  # On exclut la couche du joueur actif
                # next_state = next_state.permute(0, 3, 1, 2)
                
                # Stocker l'expérience
                action_idx = -1
                for i in range(2592):  # Nombre total d'actions possibles
                    if traduce_output(i) == action:
                        action_idx = i
                        break
                
                if action_idx != -1:
                    memory.push(state.cpu(), action_idx, reward, next_state.cpu(), done)
                
                # Mettre à jour l'état actuel
                current_player = player2
                
            else:  # Tour de l'adversaire (joueur aléatoire)
                action = ia.random_play(board, 0, current_player)
                if not action:
                    # print("Aucune action valide disponible pour l'adversaire, partie nulle.")
                    reward = 0
                    done = True
                    draws += 1
                    continue
                
                # Appliquer action
                totem_coord = model.convert_coord(action[1:3])
                totem = 'T_'+action[0]
                model.move_totem(board, totem, totem_coord)
                coord_action = model.convert_coord(action[3:])
                board[coord_action[0]][coord_action[1]] = current_player[0] + '_' + action[0]
                
                # Vérifier si la partie est terminée
                nb_free_cells = len(model.all_free_cells(board))
                if model.is_winner(board, current_player, coord_action):
                    done = True
                    losses += 1
                    moves_nb = (34 - nb_free_cells) // 2
                    reward = -1.0 + moves_nb / 16 * 0.2

                elif nb_free_cells == 2:
                    # print("Aucune action valide disponible, partie nulle.")
                    reward = 0
                    done = True
                    draws += 1
                    continue
                
                current_player = player1
            
            total_steps += 1
            
            # Entrainement du réseau si on a assez d'expériences
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.cat(states).to(device)
                # states = states.permute(0, 3, 1, 2)
                actions = torch.tensor(actions, dtype=torch.int64).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.cat(next_states).to(device)
                # next_states = next_states.permute(0, 3, 1, 2)
                dones = torch.tensor(dones, dtype=torch.bool).to(device)
                
                # Calcul des valeurs Q actuelles
                q_values = network(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Calcul des valeurs Q cibles
                with torch.no_grad():
                    next_q_values = network(next_states)
                    max_next_q = next_q_values.max(1)[0]
                    target_q_values = rewards + gamma * max_next_q * (~dones)
                
                # Calcul de la perte
                loss = F.smooth_l1_loss(q_values, target_q_values)
                
                # Optimisation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)  # Clipper les gradients
                optimizer.step()
        
        # Mise à jour d'epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Sauvegarde périodique du modèle
        if (episode + 1) % save_interval == 0:
            write_CNN(network, save_path)
            print(f"Modèle sauvegardé à l'épisode {episode+1}")
        
        # Évaluation périodique
        if (episode + 1) % eval_interval == 0:
            win_rate = wins / (wins + losses + draws) if (wins + losses + draws) > 0 else 0
            winrate.append(evaluate_model("", eval_number, network))
            print(f"Épisode {episode+1}: Victoires: {wins}, Défaites: {losses}, Nuls: {draws}, Taux de victoire: {win_rate:.4f}")
            # Réinitialisation des compteurs pour la prochaine période d'évaluation
            wins, losses, draws = 0, 0, 0
    # Sauvegarde finale du modèle
    write_CNN(network, save_path)
    print(f"Entraînement terminé. Modèle final sauvegardé à {save_path}")
    return network, winrate

def evaluate_model(model_path, num_games=100, cnn = ""):
    """
    Évalue les performances du modèle DQN contre un joueur aléatoire
    
    Args
        model_path: Chemin vers le modèle sauvegardé
        num_games: Nombre de parties à jouer pour l'évaluation
    
    Returns:
        Taux de victoire du modèle
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(cnn == ""): network = load_CNN(model_path)
    else: network = cnn
    network.to(device)
    network.eval()
    
    wins = 0
    losses = 0
    draws = 0
    
    for game in range(num_games):
        board = model.init_board()
        player1 = "DQN"
        player2 = "Random"
        current_player = player1 if game % 2 == 0 else player2  # Alterner qui commence
        done = False
        
        while not done:
            if current_player == player1:  # Tour du DQN
                state_tensor = convert_board(board, player1, player2, current_player)
                state = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
                # state = state[:, :, :, :6]
                # state = state.permute(0, 3, 1, 2)
                
                with torch.no_grad():
                    q_values = network(state).squeeze(0)
                
                action_table = filter_outputs(q_values, board, current_player)
                if not action_table[0]:
                    draws += 1
                    break
                
                best_idx = torch.argmax(torch.tensor(action_table[1])).item()
                action = action_table[0][best_idx]

                # Applique l'action
                totem_coord = model.convert_coord(action[1:3])
                totem = 'T_'+action[0]
                model.move_totem(board, totem, totem_coord)
                coord_action = model.convert_coord(action[3:])
                board[coord_action[0]][coord_action[1]] = current_player[0] + '_' + action[0]
                
                if model.is_winner(board, current_player, coord_action):
                    done = True
                    wins += 1
                
                current_player = player2
                
            else:  # Tour du joueur aléatoire
                action = ia.random_play(board, 0, current_player)
                if not action:
                    draws += 1
                    break
                
                # Applique l'action
                totem_coord = model.convert_coord(action[1:3])
                totem = 'T_'+action[0]
                model.move_totem(board, totem, totem_coord)
                coord_action = model.convert_coord(action[3:])
                board[coord_action[0]][coord_action[1]] = current_player[0] + '_' + action[0]
                
                if model.is_winner(board, current_player, coord_action):
                    done = True
                    losses += 1

                current_player = player1
    
    win_rate = wins / num_games
    print(f"Évaluation sur {num_games} parties:")
    print(f"Victoires: {wins}, Défaites: {losses}, Nuls: {draws}")
    print(f"Taux de victoire: {win_rate:.4f}")
    
    return win_rate

if __name__ == "__main__":
    # Exemple d'utilisation
    model_path = "xonox_network.bbl"
    
    # Entraînement du modèle
    trained_network, winrate = train_dqn(
        num_episodes=200,
        batch_size=32,
        gamma=0.99,
        epsilon_start=1.0, 
        epsilon_end=0.1,
        epsilon_decay=0.995,
        learning_rate=0.001,
        save_path=model_path,
        eval_interval=10,
        eval_number=50
    )
    plt.plot(winrate, label = "taux de victoire contre l'aléatoire")
    plt.legend()
    plt.show()
    # Évaluation du modèle
    evaluate_model(model_path, num_games=100)