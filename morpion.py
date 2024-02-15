import csv
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.backend import reshape
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt



def initBoard():
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    return board
coord ={
    1: [0,0],
    2: [0,1],
    3: [0,2],
    4: [1,0],
    5: [1,1],
    6: [1,2],
    7: [2,0],
    8: [2,1],
    9: [2,2],
}

# Imprime l'état actuel du plateau
def printBoard(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            mark = ' '
            if board[i][j] == 1:
                mark = 'X'
            elif board[i][j] == 2:
                mark = 'O'
            if (j == len(board[i]) - 1):
                print(mark)
            else:
                print(str(mark) + "|", end='')
        if (i < len(board) - 1):
            print("-----")


# Obtient une liste de coups valides pour un plateau donné
def getMoves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                moves.append((i, j))
    return moves


def getWinner(board):
    candidate = 0
    won = 0

    # Vérifie les rangées
    for i in range(len(board)):
        candidate = 0
        for j in range(len(board[i])):

            # S'assurer qu'il n'y a pas de vide
            if board[i][j] == 0:
                break

            # Identifie le candidat en tête
            if candidate == 0:
                candidate = board[i][j]

            # Cette condition vérifie si la colonne actuelle de la rangées est la dernière colonne de la rangées. S'il s'agit de la dernière colonne, cela signifie que le candidat en tête a occupé tous les emplacements de la rangée, et donc qu'il a gagné la partie.
            if candidate != board[i][j]:
                break
            elif j == len(board[i]) - 1:
                won = candidate

    if won > 0:
        return won

    # Vérifie les colonnes
    for j in range(len(board[0])):
        candidate = 0
        for i in range(len(board)):

            if board[i][j] == 0:
                break

            if candidate == 0:
                candidate = board[i][j]

            if candidate != board[i][j]:
                break
            elif i == len(board) - 1:
                won = candidate

    if won > 0:
        return won

    # Vérifie les diagonales
    candidate = 0
    for i in range(len(board)):
        if board[i][i] == 0:
            break
        if candidate == 0:
            candidate = board[i][i]
        if candidate != board[i][i]:
            break
        elif i == len(board) - 1:
            won = candidate

    if won > 0:
        return won

    candidate = 0
    for i in range(len(board)):
        if board[i][2 - i] == 0:
            break
        if candidate == 0:
            candidate = board[i][2 - i]
        if candidate != board[i][2 - i]:
            break
        elif i == len(board) - 1:
            won = candidate

    if won > 0:
        return won

    # Toujours pas de gagnant ?
    if (len(getMoves(board)) == 0):
        # C'est un match nul
        return 0
    else:
        # Il reste encore des mouvements à faire
        return -1


# random.seed()


# Cette fonction bestMove prend en compte l'état actuel du plateau de Tic-Tac-Toe (board), un réseau neuronal entraîné (model), le symbole du joueur (player) qui souhaite jouer un coup, et un facteur aléatoire (rnd).
def bestMove(board, model, player, rnd=0):
    scores = []
    moves = getMoves(board)

    # La fonction récupère d'abord tous les mouvements possibles du plateau actuel à l'aide de la fonction getMoves.
    # Ensuite, pour chaque mouvement possible, elle crée un nouveau plateau (future) en plaçant le mouvement du joueur sur le plateau et prédit le résultat à l'aide du modèle fourni
    for i in range(len(moves)):
        future = np.array(board)
        future[moves[i][0]][moves[i][1]] = player
        # future.reshape((-1, 9)) est utilisé pour remodeler le tableau future, qui représente un état futur possible du tableau, en une forme compatible avec la forme d'entrée du modèle. Le -1 dans la commande reshape signifie que la taille de cette dimension doit être déduite en fonction de la taille des autres dimensions, et le 9 est le nombre total de cellules dans la grille de tic-tac-toe.
        # model.predict est ensuite utilisé pour effectuer une prédiction sur le tableau futur remodelé. Il renvoie un tableau de forme (1, 3), chaque élément du tableau représentant la probabilité prédite par le modèle d'un match nul, d'une victoire pour le joueur 1 ou d'une victoire pour le joueur 2, respectivement.
        # Enfin, [0] est utilisé pour extraire le premier (et unique) élément de ce tableau, qui représente la prédiction du modèle pour l'état donné du tableau.
        prediction = model.predict(future.reshape((-1, 9)), verbose=0)[0]
        # Les valeurs prédites sont ensuite utilisées pour calculer un score pour chaque coup, qui est ajouté à une liste de scores (scores).
        if player == 1:
            winPrediction = prediction[1]
            lossPrediction = prediction[2]
        else:
            winPrediction = prediction[2]
            lossPrediction = prediction[1]
        drawPrediction = prediction[0]
        # Les scores sont calculés sur la base de la différence entre les valeurs prédites pour gagner et perdre (winPrediction et lossPrediction) ou la valeur prédite pour faire match nul (drawPrediction).
        if winPrediction - lossPrediction > 0:
            scores.append(winPrediction - lossPrediction)
        else:
            scores.append(drawPrediction - lossPrediction)

    # La fonction trie ensuite les scores par ordre décroissant et sélectionne au hasard le meilleur coup avec une certaine probabilité (rnd) ajoutée pour introduire un certain niveau d'aléa dans la décision.
    # Si le facteur aléatoire est fixé à 0, la fonction sélectionne toujours le meilleur coup. S'il existe plusieurs meilleurs coups, la fonction en choisit un au hasard. Si aucun coup valide n'est disponible, la fonction renvoie None.
    bestMoves = np.flip(np.argsort(scores))
    for i in range(len(bestMoves)):
        if random.random() * rnd < 0.5:
            return moves[bestMoves[i]]

    return moves[random.randint(0, len(moves) - 1)]


# La fonction simulateGame simule une partie de morpion entre deux joueurs. Elle prend trois paramètres :
# p1 : un modèle de joueur pour le joueur 1, qui est utilisé pour déterminer le meilleur mouvement pour le joueur 1.
# p2 : un modèle de joueur pour le joueur 2, qui est utilisé pour déterminer le meilleur mouvement pour le joueur 2.
# rnd : un facteur aléatoire utilisé pour choisir le meilleur coup, qui détermine la pondération des scores prédits par le modèle par rapport à un coup aléatoire.
def simulateGame(p1=None, p2=None, rnd=0):
    history = []
    board = initBoard()
    playerToMove = 1
    # La fonction initialise un tableau et une variable playerToMove qui indique à qui revient le tour de se déplacer. Elle entre ensuite dans une boucle qui s'exécute jusqu'à ce qu'un gagnant soit trouvé (comme déterminé par la fonction getWinner).
    while getWinner(board) == -1:

        # Dans la boucle, la fonction choisit un coup pour le joueur actuel en utilisant la fonction bestMove. Si un modèle de joueur est fourni pour le joueur actuel, bestMove est appelé avec ce modèle et le numéro du joueur actuel. Si aucun modèle de joueur n'est fourni, un mouvement aléatoire est choisi.
        move = None
        if playerToMove == 1 and p1 != None:
            move = bestMove(board, p1, playerToMove, rnd)
        elif playerToMove == 2 and p2 != None:
            move = bestMove(board, p2, playerToMove, rnd)
        else:
            moves = getMoves(board)
            move = moves[random.randint(0, len(moves) - 1)]

        # Le mouvement choisi est ensuite exécuté sur le plateau et ajouté à l'historique du jeu.
        board[move[0]][move[1]] = playerToMove

        history.append((playerToMove, move))

        # le joueur actif est changé.
        playerToMove = 1 if playerToMove == 2 else 2

    return history


# Simulation d'un jeu
# history = simulateGame()
# Nous voyons que l'"history" d'un jeu consiste en un tableau de tuples. Le premier élément de chacun de ces tuples est le joueur qui s'est déplacé (1 pour 'X', 2 pour 'O'),
# print(history)

# Reconstruit le tableau à partir de la liste des mouvements
def movesToBoard(moves):
    board = initBoard()
    for move in moves:
        player = move[0]
        coords = move[1]
        board[coords[0]][coords[1]] = player
    return board


# board = movesToBoard(history)
# printBoard(board)
# Désormais, lorsque nous transmettons à cette fonction l'historique d'un jeu, nous pouvons voir le tableau qui en résulte.
# print(getWinner(board))

# Simule des jeux aléatoires, produisons des statistiques globales pour 10 000 jeux à la fois.
# games = [simulateGame() for _ in range(10000)]

# Prend en entrée une liste de jeux et un numéro de joueur et renvoie les statistiques des performances du joueur dans ces jeux.
def gameStats(games, player=1, filename='game_stats.csv'):
    stats = {"win": 0, "loss": 0, "draw": 0}
    # Pour chaque jeu de la liste d'input,on utilise la fonction getWinner pour déterminer le résultat du jeu.
    for game in games:
        result = getWinner(movesToBoard(game))
        # Si le résultat est -1 (indiquant que le jeu est toujours en cours), la fonction passe au jeu suivant.
        if result == -1:
            continue
        # Si le résultat est égal au numéro de "player", la fonction incrémente le nombre de "win" dans les statistiques.
        elif result == player:
            stats["win"] += 1
        # Si le résultat est egal a zero (ce qui indique un match nul), la fonction incrémente le nombre de "draw".
        elif result == 0:
            stats["draw"] += 1
        # Sinon, la fonction incrémente le nombre de "loss".
        else:
            stats["loss"] += 1

    # Après avoir compté le nombre de victoires, de défaites et de match nuls pour le joueur dans la liste de "games", la fonction calcule le pourcentage de victoires, le pourcentage de défaites et le pourcentage de match nuls.
    winPct = stats["win"] / len(games) * 100
    lossPct = stats["loss"] / len(games) * 100
    drawPct = stats["draw"] / len(games) * 100

    # Écriture des résultats dans le fichier CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Win %", "Loss %", "Draw %"])
        writer.writerow([winPct, lossPct, drawPct])



# gameStats(games,filename="game_stats_rnd.csv")
# print()
# Appliquer cette fonction aux jeux que nous avons générés, afin d'observer ce qui se passe dans un jeu complètement aléatoire.
# gameStats(games, player=2)

# Renvoie un modèle de réseau neuronal compilé. Il s'agit d'un réseau neuronal feedforward composé de quatre couches "Dense", de deux couches "Dropout" et d'une couche d'ouput "softmax".
def getModel():
    # La couche d'input comporte numCells, des neurones d'input, chacun correspondant à une cellule d'un tableau de morpion.
    numCells = 9
    # La couche d'ouput comporte des neurones de sortie (3 dans le cas présent), chacun correspondant à un résultat possible d'une partie : une victoire pour le joueur 1, une victoire pour le joueur 2 ou un match nul.
    outcomes = 3
    # Crée un nouvel objet de modèle séquentiel Keras, qui est une pile linéaire de couches de réseau neuronal.
    model = Sequential()
    # La première couche Dense compte 200 neurones et utilise la fonction d'activation ReLU (unité linéaire rectifiée).
    model.add(Dense(200, activation='relu', input_shape=(9,)))
    # La deuxième couche est une couche Dropout, qui élimine de manière aléatoire 20 % des neurones afin d'éviter un surajustement.
    model.add(Dropout(0.2))
    # Les troisième et quatrième couches Dense comptent respectivement 125 et 75 neurones et utilisent l'activation ReLU.
    model.add(Dense(125, activation='relu'))
    model.add(Dense(75, activation='relu'))
    # La cinquième couche est une autre couche Dropout qui exclut 10 % des neurones.
    model.add(Dropout(0.1))
    # La dernière couche compte 25 neurones et utilise l'activation ReLU.
    model.add(Dense(25, activation='relu'))
    # La couche d'ouput comporte des neurones d'outcomes,3 neurones (un pour chaque résultat) et une fonction d'activation softmax, qui produit une distribution de probabilité sur les résultats possibles.
    model.add(Dense(outcomes, activation='softmax'))
    # Le modèle est compilé avec la fonction de perte categorical_crossentropy, l'optimiseur rmsprop et la métrique de précision acc. C'est ainsi que nous indiquons au modèle que nous voulons qu'il produise un tableau de probabilités. Ces probabilités indiquent la confiance du modèle dans chacun des trois résultats du jeu pour un plateau donné.
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


# model = getModel()
# print(model.summary())

# Reçoit une liste de jeux de morpion et les convertit en données adaptées à l'entraînement d'un modèle de machine learning pour prédire les résultats des victoires et des défaites.
def gamesToWinLossData(games):
    X = []
    y = []
    # Utilise la fonction movesToBoard pour convertir chaque coup d'un jeu en un état du plateau et rassemble tous les états du plateau et leurs résultats correspondants (victoire ou défaite) dans des listes distinctes X et y.
    for game in games:
        winner = getWinner(movesToBoard(game))
        for move in range(len(game)):
            X.append(movesToBoard(game[:(move + 1)]))
            y.append(winner)
    # Les listes X et y résultantes sont ensuite remodelées et converties en tableaux numpy. Le tableau y est codé à une clé à l'aide de la fonction to_categorical de Keras, qui convertit les étiquettes de résultat (victoire ou perte) en un vecteur binaire.
    X = np.array(X).reshape((-1, 9))
    y = to_categorical(y)

    # Divise les données en une division train/valid avec 80 % des données utilisées pour la formation et 20 % pour la validation.
    trainNum = int(len(X) * 0.8)
    return (X[:trainNum], X[trainNum:], y[:trainNum], y[trainNum:])

model = load_model('my_model.h5')
# model = getModel()
# games = [simulateGame(p2=model) for _ in range(1000)]
# Séparer les données de formation et de validation
# X_train, X_test, y_train, y_test = gamesToWinLossData(games)

# nEpochs spécifie le nombre de fois que l'ensemble des données sera itéré pendant la formation
# nEpochs = 1000

# batchSize spécifie le nombre d'échantillons qui seront propagés à travers le réseau en une seule fois.
# batchSize = 100

# X_train et y_train sont les données d'input et d'ouput pour la formation, et X_test et y_test pour la validation.
# La fonction "fit" entraîne ensuite le modèle pour le nombre d'epochs et le batch_size spécifiés, en mettant à jour les paramètres du modèle pour minimiser la fonction de perte à l'aide de l'optimiseur spécifié dans la fonction de compilation.
# Pendant l'apprentissage, la fonction fit affiche sur la console les mesures de perte et de précision d'apprentissage et de validation pour chaque epochs.
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nEpochs, batch_size=batchSize)
# gameStats(games, player=2)
# model = load_model('my_model.h5')
# model.save('my_model.h5')



# Ensuite, nous utilisons ce modèle entraîné pour jouer à un jeu et voir si nous faisons mieux que le hasard.
# games2 = [simulateGame(p1=model) for _ in range(1000)]
# gameStats(games2)


# games3 = [simulateGame(p2=model) for _ in range(1000)]
# gameStats(games3, player=2)
# for i in range(len(games3)):
#     board = movesToBoard(games3[i])
#     printBoard(board)
#     print(getWinner(board))


def printBoardTkinter(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 1:
                buttons[i][j].config(text="X")
            elif board[i][j] == 2:
                buttons[i][j].config(text="O")


def button_click(row, col):
    root.after(0, Game(board, row, col, p2= model))
    printBoardTkinter(board)
    print(board)
    return row, col


root = tk.Tk()
root.title("Tic Tac Toe")
board = initBoard()
buttons = []

for row in range(3):
    button_row = []
    for col in range(3):
        button = tk.Button(root, text="", width=10, height=5,
                           command=lambda row=row, col=col: button_click(row, col))
        button.grid(row=row, column=col)
        button_row.append(button)
    buttons.append(button_row)

def save_history(history):
    df = pd.DataFrame(history, columns=['player', 'move'])
    df.to_csv('history.csv',mode='a', index=False)


def Game(board, p1=None, p2=None, rnd=0):
    history = []
    playerToMove = 1

    while getWinner(board) == -1:
        printBoard(board)

        if playerToMove == 1:
            print("Player 1's turn (X)")
            while True:
                try:
                    ans= int(input("Enter a number (1-8): "))
                    row = coord[ans][0]
                    col = coord [ans][1]
                    if board[row][col] == 0:
                        break
                    else:
                        print("Wrong move a$$hole.")
                except ValueError:
                    print("Invalid input. I said a NUMBER you dog c*nT.")
                except KeyError:
                    print("Please input a valid number dumba$$")

            move = (row, col)
        else:
            move = bestMove(board, p2, playerToMove, rnd)

        board[move[0]][move[1]] = playerToMove
        history.append((playerToMove, move))

        playerToMove = 1 if playerToMove == 2 else 2

    printBoard(board)
    save_history(history)


def BarStats(filename='game_stats.csv'):
    df = pd.read_csv(filename)
    # Trace le diagramme en barres
    ax = df.plot(kind='bar', rot=0)
    ax.set_title("Game Statistics")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count / %")
    ax.set_ylim([0, 100])
    ax.legend(loc='best')

    # Affiche le diagramme
    plt.show()

# BarStats(filename="game_stats_rnd.csv")
# root.mainloop()
# printBoard(board)


Game(board, p2=model)
