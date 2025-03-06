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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    
    """
    def __init__(self, depth='2'):
        self.depth = int(depth)


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsules = currentGameState.getCapsules()
        if capsules:
            nearestCapsuleDistance = min([manhattanDistance(newPos, capsule) for capsule in capsules])
        else:
            nearestCapsuleDistance = 1 
        capsuleScore = 1.0 / nearestCapsuleDistance if nearestCapsuleDistance else 0
        food = newFood.asList() 
        if food:
            nearest_food = min([manhattanDistance(newPos, f) for f in food])
        else: nearest_food = 1
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        nearest_ghost = min([manhattanDistance(newPos, g) for g in ghostPositions])
        if nearest_food!= 0:
            food_points = 1.0/ nearest_food
        else:
            food_points = float('inf')
        if nearest_ghost!= 0:
            ghost_points = -1.0/ nearest_ghost
        else:
            ghost_points = -float('inf')
        scaredGhostScore = 0
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                scaredGhostScore += 1.0 / ghostDistance if ghostDistance != 0 else float('inf')
        stopPenalty = -10 if action == Directions.STOP else 0

        return successorGameState.getScore() + food_points + ghost_points + scaredGhostScore + stopPenalty

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
    
        def minimize(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                evalScore = self.evaluationFunction(state)
                return evalScore

            best = float("inf")
            legalActions = state.getLegalActions(agentIndex)
            numGhosts = state.getNumAgents()

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == numGhosts - 1:
                    best = min(best, maximize(successor, depth + 1))
                else: 
                    best = min(best, minimize(successor, depth, agentIndex + 1))
            return best

        def maximize(state, depth):
            print(depth)
            if state.isWin() or state.isLose() or depth == self.depth:
                evalScore = self.evaluationFunction(state)
               
                return evalScore

            best_val = -float("inf")
            print(state.getLegalActions(0))
            legalActions = state.getLegalActions(0)

            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                best_val = max(best_val, minimize(successor, depth, 1))
            print(f"Our bestest val at depth -> {depth}: {best_val}")
            return best_val

        bestAction = None
        bestValue = -float("inf")
        legalActions = gameState.getLegalActions(0)

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minimize(successor, 0, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        print(f"final action: {bestAction}, {bestValue}")
        return bestAction
            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        infinity = float('inf')
        neg_infinity = -float('inf')

        def maximize(state, depth, alpha, beta):
            if state.isWin() or state.isLose(): 
                return self.evaluationFunction(state)
            val, bestAction = neg_infinity, Directions.STOP
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                currVal = minimize(successor, depth, 1, alpha, beta)
                if currVal > val: val, bestAction = currVal, action
                if val > beta: 
                    return val
                alpha = max(alpha, val)
            return bestAction if depth == 1 else val

        def minimize(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose(): 
                return self.evaluationFunction(state)
            val = infinity
            numAgents = state.getNumAgents()
            nextAgentIndex = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                currVal = (maximize(successor, nextDepth, alpha, beta) if nextAgentIndex == 0 and nextDepth <= self.depth else
                           self.evaluationFunction(successor) if nextAgentIndex == 0 else
                           minimize(successor, nextDepth, nextAgentIndex, alpha, beta))
                val = min(val, currVal)
                if val < alpha: 
                    return val
                beta = min(beta, val)
            return val

        return maximize(gameState, 1, neg_infinity, infinity)
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """

        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth or not state.getLegalActions(agentIndex):
                return self.evaluationFunction(state)
            print(gameState.getLegalActions(0))

            legalActions = state.getLegalActions(agentIndex)

            if agentIndex == 0:
                print(gameState.getLegalActions(0))
                return max(expectimax(state.generateSuccessor(agentIndex, action), depth, 1) for action in legalActions)
            else: 
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return sum(expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in legalActions) / len(legalActions)

        bestAction = None
        bestValue = -float("inf")
        legalActions = gameState.getLegalActions(0)
        print(gameState.getLegalActions(0))

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 0, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
        


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <gets pos of food pellets, ghosts, scared time remaining, and capsule positions>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()

    # Calculate dis to the nearest capsule
    if capsules:
        nearestCapsuleDistance = min([manhattanDistance(newPos, capsule) for capsule in capsules])
    else:
        nearestCapsuleDistance = 1 
    capsuleScore = 1.0 / nearestCapsuleDistance if nearestCapsuleDistance else 0

    # Calculate dis to the nearest food pellet
    food = newFood.asList() 
    if food:
        nearestFoodDistance = min([manhattanDistance(newPos, f) for f in food])
    else:
        nearestFoodDistance = 1
    foodScore = 1.0 / nearestFoodDistance if nearestFoodDistance else float('inf')

    # Calculate dist to the nearest ghost
    ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
    nearestGhostDistance = min([manhattanDistance(newPos, g) for g in ghostPositions])
    ghostScore = -1.0 / nearestGhostDistance if nearestGhostDistance else -float('inf')
    # Calculate the score por scared ghosts
    scaredGhostScore = 0
    for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(newPos, ghostPos)
        if scaredTime > 0:
            scaredGhostScore -= 1.0 / ghostDistance if ghostDistance != 0 else float('inf')
    x,y = newPos
    curr_food = 0
    curr_wall = 0
    if currentGameState.hasFood(x,y):
        curr_food = 3
    
    if currentGameState.hasWall(x,y):
        curr_wall = -3
    # stopping = bad

    # combine score for final heuristic
    return currentGameState.getScore() + foodScore + ghostScore + scaredGhostScore + capsuleScore + curr_wall + curr_food

    

# Abbreviation
better = betterEvaluationFunction
