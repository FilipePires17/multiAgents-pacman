# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from audioop import minmax
from util import manhattanDistance
from game import Directions
import random
import util
import math
import numpy as np

from game import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, depth, agent=1):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()
        if agent == 0:
            val = -math.inf
            actions = gameState.getLegalPacmanActions()
            for action in actions:
                newState = gameState.generatePacmanSuccessor(action)
                val = max(val, self.minimax(newState, depth - 1, 1))
            return val
        else:
            val = math.inf
            actions = gameState.getLegalActions(agent)
            newAgent = agent + 1 if agent + 1 < numAgents else 0
            for action in actions:
                newState = gameState.generateSuccessor(agent, action)
                val = min(val, self.minimax(newState, depth - 1, newAgent))
            return val

            distanciasAntes = []
            ghostPositionsAntes = gameState.getGhostPositions()
            for pos in ghostPositionsAntes:
                distanciasAntes.append(
                    manhattanDistance(pos, gameState.getPacmanPosition())
                )
            acoesFantasma = ["" for _ in range(len(ghostPositionsAntes))]
            for i in range(len(ghostPositionsAntes)):
                actions = gameState.getLegalActions(i + 1)
                achou = False
                for action in actions:
                    newState = gameState.generateSuccessor(i + 1, action)
                    posF = newState.getGhostPosition(i + 1)
                    distancia = manhattanDistance(posF, newState.getPacmanPosition())
                    if not achou:
                        acoesFantasma[i] = action
                    if distancia < distanciasAntes[i]:
                        acoesFantasma[i] = action
                        achou = True

            newState = gameState
            for i in range(len(acoesFantasma)):
                if newState.isLose():
                    return self.evaluationFunction(newState)
                newState = newState.generateSuccessor(i + 1, acoesFantasma[i])

            val = self.minimax(newState, depth - 1, True)

            return val

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        actions = gameState.getLegalPacmanActions()
        teste = []
        for action in actions:
            newState = gameState.generatePacmanSuccessor(action)
            value = self.minimax(newState, self.depth)
            teste.append(value)
        # print(teste)
        max_val = max(teste)
        # print(max_val)
        ind = teste.index(max_val)
        # print(actions)
        # print(actions[ind])
        return actions[ind]
        # esquerda, parado, direita, cima, baixo


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, gameState, depth, a, b, agent=1):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()
        if agent == 0:
            val = -math.inf
            actions = gameState.getLegalPacmanActions()
            for action in actions:
                newState = gameState.generatePacmanSuccessor(action)
                val = max(val, self.alphabeta(newState, depth - 1, a, b, 1))
                a = max(a, val)
                if b <= a:
                    break
            return val
        else:
            val = math.inf
            actions = gameState.getLegalActions(agent)
            newAgent = agent + 1 if agent + 1 < numAgents else 0
            for action in actions:
                newState = gameState.generateSuccessor(agent, action)
                val = min(val, self.alphabeta(newState, depth - 1, a, b, newAgent))
                b = min(val, a)
                if b < a:
                    break
            return val

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = gameState.getLegalPacmanActions()
        teste = []
        for action in actions:
            newState = gameState.generatePacmanSuccessor(action)
            value = self.alphabeta(newState, self.depth, -math.inf, math.inf)
            teste.append(value)
        max_val = max(teste)
        ind = teste.index(max_val)

        return actions[ind]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
