Imagine a game of chess where you are playing against an opponent. You want to win the game, however you don't know what moves your opponent will make and which moves will lead to a win.
 
 ## How can we win the game?
 
 This is where the minimax algorithm comes in. The minimax algorithm is a decision-making algorithm that is used in two-player games to determine the best move for a player. It works by considering all possible moves that a player can make and then choosing the move that will lead to the best outcome for the player. The minimax algorithm is based on the assumption that the opponent will also make the best move possible, and it aims to minimize the maximum possible loss for the player.

 ## Lets define a game tree.

 For the calculation, we need to start by defining a game tree, where each node represents a possible state of the game and each edge represents a possible move the player can make.

 ![Game Tree](/blog/images/game-tree.png)

 For chess this would look like the following, where each node represents a state of the board and each edge represents a possible move that can be made. In this example two possible next states would be that the white player takes the pawn on e5 with the Knight or progresses the pawn on d2 to d4, threatening the black pawn on e5.

![Chess Game Tree](/blog/images/game-tree-chess.png)

## How do we decide which move to make?

The minimax algorithm works by evaluating all possible move sequences up to a certain depth in the game tree and then choosing the move that leads to the best outcome for the player.

## Why do we need to restrict the depth?
Note that we can't simulate all possible moves to the end of the game, as the number of possible moves grows exponentially with the depth of the game tree. Specifically, O(b^d) where b is the branching factor (number of possible moves in each state) and d is the depth of the tree. Therefore we first fix the depth of the tree to a certain value and then the algorithm starts evaluating the game tree from the bottom up.

## First Step: Evaluate the leaf nodes.
We first need to evaluate the values of the leaf nodes of the game tree. This is done by a heuristic function that assigns a value to each node based on how good the state of the game is for the player. For example, in chess, we could assign a value to each state based on the number of pieces each player has, the position of the pieces, and the number of possible moves each player has.

![Minimax Step 1](/blog/images/minimax-step-1.png)

## Second Step: Propagate the values up the tree.
Once we have evaluated the leaf nodes, we can propagate the values up the tree. For each node, we choose the move that leads to the best outcome for the player. If the player is the maximizing player, we choose the move that leads to the highest value, and if the player is the minimizing player, we choose the move that leads to the lowest value. In our example there is the maximizing player first, so we choose the maximum value of the children.

![Minimax Step 2](/blog/images/minimax-step-2.png)

Since the previous step was for the maximizing player, now its the turn for the minimizing player (the opponent). He will choose the move that leads to the lowest value for the current player.

![Minimax Step 3](/blog/images/minimax-step-3.png)

## Third Step: Choose the best move.
Since we are now at the root node, we can choose the move that leads to the best outcome for us. In this case, we would choose the move that leads to the highest value.

![Minimax Step 4](/blog/images/minimax-step-4.png)

## Code Implementation.
Now that we know how this algorithm works, let's implement it in Python. We will create a simple game tree with nodes that have two children each and a random value assigned to them if evaluated. For simplicity we will not have terminal nodes in this example - however you could adjust the code to sample if a node is terminal or not on creation.

```python
import random
import math

# Counter to show-case the creation of Nodes
# Note this grows with O(2^depth) and we always have root node.
creation_counter = 1

# This is a simple Node class that represents a node in the game tree.
# Each node gets two children (two possible moves) and if evaluated gets a random value. 
# There are no terminal nodes in this example.
class Node:
    def __init__(self):
        self._children = None
        self.value = None

    @property
    def children(self):
        # Two Children for each Node
        if self._children is None:
            global creation_counter
            creation_counter += 2
            self._children = [Node() for _ in range(2)]
        return self._children

    def evaluate(self):
        # Random Value for the Node if Evaluation is needed
        if self.value is None:
            self.value = random.randint(0, 100)
        return self.value

    def is_terminal(self):
        # In this example no Node is terminal
        return False

# Core Algorithm
# Recursively evaluates the game tree and returns the value of the best move.
def minimax(node, depth, maximizingPlayer):
    if depth == 0 or node.is_terminal():
        value = node.evaluate()
        node.value = value
        return value

    if maximizingPlayer:
        value = -math.inf
        for child in node.children:
            value = max(value, minimax(child, depth - 1, False))
        node.value = value
        return value
    else:
        value = math.inf
        for child in node.children:
            value = min(value, minimax(child, depth - 1, True))
        node.value = value
        return value

# Main Function
# Test here with different depths and see how the computation time and number of nodes created grows.
if __name__ == "__main__":
    start_node = Node()
    depth = 3
    minimax_value = minimax(start_node, depth=depth, maximizingPlayer=True)

    print(f'Found Best Action with Value: {minimax_value}')
    if start_node._children is not None:
        # Avoid printing the children if they are not created
        print(f'Child Left: {start_node.children[0].value}')
        print(f'Child Right: {start_node.children[1].value}')
    print(f'Number of Nodes Created: {creation_counter}')

    # Result - Depth 3
    # Found Best Action with Value: 66
    # Child Left: 44
    # Child Right: 66
    # Number of Nodes Created: 15
```
Try to adjust the depth and see how the number of nodes created grows exponentially. The more nodes are created the more computation time is needed to evaluate the game tree. 

## Can we improve?
While the minimax algorithm is a powerful tool for decision-making in two-player games, it has some limitations. One of the main limitations is that it can be computationally expensive, especially for games with a large branching factor and depth. To address this issue, we can use the alpha-beta pruning algorithm, which is an extension of the minimax algorithm that reduces the number of nodes that need to be evaluated.

##  References
[Wikipedia - Minimax](https://en.wikipedia.org/wiki/Minimax) - Wikipedia article on Minimax.

[Chess Programming](https://www.chessprogramming.org/Minimax) - Chess Programming Wiki on Minimax.

[//]: # ([alpha-beta pruning algorithm]/#/blog/alpha-beta)