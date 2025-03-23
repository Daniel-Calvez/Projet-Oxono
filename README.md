# Projet OXONO
### Vincent Ducot et Daniel Calvez
---

## Prérequis

### Pour le jeu, les test et l'IA aléatoire

Il faut avant tout python (Développé sur la version 3.12)

https://www.python.org/downloads/

Il est nécessaire d'installer pytest et argparse

<code>pip install pytest<br>
pip install icecream<br>
pip install argparse</code>

### Pour l'IA basée sur le Deep-Q-Learning

Il est necessaire d'installer pytorch en plus

<code>pip install pytorch<br>
pip install matplotlib<br>
pip install pathlib</code>

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
