# Projet OXONO
### Vincent Ducot et Daniel Calvez
---

## Prérequis

Il faut avant tout python (Développé sur la version 3.12)

https://www.python.org/downloads/

Le setup.py installe les packages nécessaires.
<code>python setup.py install</code>

## Utilisation
usage: python game.py [-h] [--player1-ia] [--player2-ia] [--ia1-level {0,1,2}] [--ia2-level {0,1,2}] player1 player2

positional arguments:
  player1              Nom du 1er joueur (ou IA)
  player2              Nom du 2e joueur. Si le second joueur est une IA, elle sera nommée automatiquement.

options:
  -h, --help           show this help message and exit
  --player1-ia         Si le 1er joueur est une IA.
  --player2-ia         Si le 2e joueur est une IA.
  --ia1-level {0,1,2}  Niveau de la 1ère IA : 0 (random), 1 (random++), 2 (DQN)
  --ia2-level {0,1,2}  Niveau de la 2e IA : 0 (random), 1 (random++), 2 (DQN)

### Exemples
Ppour jouer contre l'IA
`python game.py --player2-ia --ia2-level 2 Guillaume IA`

Pour faire s'affronter 2 IA
`python game.py IA1 IA --player2-ia --player1-ia --ia1-level 1 --ia2-level 2`

Pour lancer les tests
`pytest .\test_model.py`

## IA
L'IA niveau 0 et 1 (mouvements aléatoires) sont dans le fichier ia.py. </br>
L'IA niveau 2 est basée sur un réseau de neurones à convolution (CNN), avec une fonction de Q learning qui assigne un score à des paire (état-action). Cette approche s'appelle le Deep-Q-Learning (DQN). </br>
Pour utiliser cette IA, il faut le fichier dqn.py (qui est appelé par ia.py) ainsi qu'un modèle entrainé.</br>
Un modèle est fourni dans le zip xonox.dqn.zip.</br>
Les fichiers dqn_random_train.py et dqn_self_train.py servent à entrainer le modèle, contre respectivement un adversaire aléatoire et contre l'IA DQN.