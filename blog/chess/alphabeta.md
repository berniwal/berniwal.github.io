In the previous post, we explored the [minimax algorithm](/#/blog/mini-max) and saw how it evaluates every branch of the game tree to determine the best move. However, the exponential growth in nodes evaluated can be a major performance issue. That's where alpha-beta pruning comes in.

Alpha-beta pruning is an enhancement to minimax that reduces the number of nodes processed in the search tree. The main idea is to stop evaluating a move when it becomes clear that it won’t influence the final decision.

## Which moves can be pruned?

If you look at the game tree below, you will notice that some branches are irrelevant to the final decision. For example, in the following tree, it would not make sense to evaluate the leaf node X. Because even if the value would be larger again, it would never be selected anyway - since our opponent will always choose the action which will lead to a lower reward of 5. Therefore, we can prune the evaluation of node X and save some computation time.

This is what we call a beta cutoff. The beta value represents the best score (lowest) that the minimizing player is assured of. If we find a move that is better than the current best assured score for our opponent, we can stop evaluating further children. In our example the lowest score our opponent can achieve so far is 5, since we have found a better score of 8 for the current route, we can stop evaluating further children, since our opponent would never choose this path.

 ![Game Tree](/blog/images/alphabeta-beta.png)

As the computation continues, we see a similar behaviour for the nodes in group Y - they will never be selected since the route we have found with reward 4 is already lower than what the maximizing player could achieve with the route on the left side - he would therefore never choose the right side.

This is what we call an alpha cutoff. The alpha value represents the best score (highest) that the maximizing player is assured of. If we find a move that is worse than the current best assured score, we can stop evaluating further children.

 ![Game Tree](/blog/images/alphabeta-alpha.png)

## Code Implementation
The implementation of the alpha-beta pruning algorithm is very similar to the minimax algorithm. The only difference is that we keep track of two values, alpha and beta, which represent the best score that the maximizing player is assured of and the best score that the minimizing player is assured of, respectively.

```python
import random
import math

# Counter to showcase node visits. Note we always visit the root.
visit_counter = 1

# Simple Node class representing a node in the game tree.
class Node:
    def __init__(self):
        self._children = None
        self.value = None

    @property
    def children(self):
        # Two Children for each Node
        if self._children is None:
            self._children = [Node() for _ in range(2)]
        return self._children

    def evaluate(self):
        # Assign a random value if the node is evaluated.
        if self.value is None:
            self.value = random.randint(0, 100)
        return self.value

    def is_terminal(self):
        # In this example no node is terminal.
        return False

# Alpha-Beta Pruning Algorithm
def alphabeta(node, depth, alpha, beta, maximizingPlayer):
    global visit_counter
    if depth == 0 or node.is_terminal():
        value = node.evaluate()
        node.value = value
        return value

    if maximizingPlayer:
        value = -math.inf
        for child in node.children:
            visit_counter += 1
            value = max(value, alphabeta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cutoff
        node.value = value
        return value
    else:
        value = math.inf
        for child in node.children:
            visit_counter += 1
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if beta <= alpha:
                break  # alpha cutoff
        node.value = value
        return value

# Main Function
if __name__ == "__main__":
    start_node = Node()
    depth = 10
    best_value = alphabeta(start_node, depth, -math.inf, math.inf, True)

    print(f'Found Best Action with Value: {best_value}')
    if start_node._children is not None:
        print(f'Child Left: {start_node.children[0].value}')
        print(f'Child Right: {start_node.children[1].value}')
    print(f'Number of Nodes Created: {visit_counter}')

    # For Depth 10, previous implementation would have created 2047 nodes.
    # Here with Alpha Beta Pruning, we get the following result:
    # Found Best Action with Value: 40
    # Child Left: 40
    # Child Right: 29
    # Number of Nodes Created: 681
```

If you compare this algorithm to the number of nodes created and the computation time of the minimax algorithm, you will see a significant improvement. You are now able to run the algorithm with a greater depth and still get a result in a reasonable amount of time. Note that this returns the same result as the minimax algorithm before, since we only remove nodes from the evaluation which would not influence the final decision anyway.

## How much did the performance improve?
In the worst case scenario, the alpha-beta pruning algorithm will still need to evaluate all nodes in the game tree - therefore O(b^d) nodes, where b is the branching factor and d is the depth of the tree. However, in the best case scenario, the algorithm will only need to evaluate O(b^(d/2)) or equivalently O(sqrt(b^d)) nodes, so it essentially halves the depth of the tree. A mathematical analysis for this scenario you will find in the references. This is a significant improvement over the O(b^d) nodes that the minimax algorithm would need to evaluate in every case.

## What’s Next?
In the next post, we will explore Monte Carlo Tree Search (MCTS), an algorithm that helped AlphaGo defeat the world champion in Go.

 ##  References
 [Wikipedia - Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) - Wikipedia article on Alpha-Beta Pruning.

 [Best-Case Analysis of Alpha-Beta Pruning](https://cw.fel.cvut.cz/b222/_media/courses/be5b33kui/labs/weekly/a-b-analysis.pdf) - Analysis of the best case scenario for alpha-beta pruning.

 [Iterative Deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search) - Algorithm to apply alpha-beta pruning as deeply as possible when facing time constraints.