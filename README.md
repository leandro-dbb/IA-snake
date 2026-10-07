# IA Snake : un Snake qui apprend à jouer tout seul

Agent de **reinforcement learning (Deep Q-Learning)** en **PyTorch** qui apprend à jouer au Snake sans aucune règle codée en dur, uniquement à partir de récompenses.

![Courbe d'apprentissage](SnakeAI/myplot.png)

## Comment ça marche

- **État (11 valeurs)** : danger devant / à droite / à gauche, direction actuelle, position relative de la nourriture.
- **Actions (3)** : tout droit, tourner à droite, tourner à gauche.
- **Récompense** : +10 quand le serpent mange, −10 quand il meurt.
- **Modèle** : réseau linéaire 11 → 256 → 3 (`Linear_Qnet`), entraîné avec l'équation de Bellman (γ = 0,9, Adam, lr = 0,001, perte MSE).
- **Mémoire de rejeu** + entraînement court terme (à chaque coup) et long terme (par batch).
- **Exploration ε-greedy** qui décroît au fil des parties.

## Lancer

```bash
pip install torch pygame numpy matplotlib
cd SnakeAI
python agent.py
```

La courbe du score et de la moyenne s'affiche en direct ; le meilleur modèle est sauvegardé dans `model/model.pth`.

## Structure

| Fichier | Rôle |
|---|---|
| `Snake_Leandro_AI.py` | Le jeu (Pygame), piloté par l'agent |
| `agent.py` | Agent : état, mémoire, ε-greedy, boucle d'entraînement |
| `model.py` | Réseau `Linear_Qnet` et `QTrainer` |
| `helper.py` | Tracé de la courbe d'apprentissage |

Version jouable à la main : [Snake-Leandro](https://github.com/leandro-dbb/Snake-Leandro).
