import random, os

class Node:
    def __init__(self, state, parent=None, move=None, depth=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = depth + misplaced_tiles(state) # Evaulation Function f(n) = Cost so far g(n) + Estimated Cost to goal h(n)
        self.statistics:str

    def __lt__(self, other):
        return self.cost < other.cost

def misplaced_tiles(state): # h(n)
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != goal_state[i][j] and state[i][j] != 0:
                count += 1
    return count

def possible_moves(state):
    moves = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                if i > 0:
                    moves.append((i - 1, j))  # Move Up
                if i < 2:
                    moves.append((i + 1, j))  # Move Down
                if j > 0:
                    moves.append((i, j - 1))  # Move Left
                if j < 2:
                    moves.append((i, j + 1))  # Move Right
                return moves

def get_new_state(state, move):
    new_state = [row[:] for row in state]
    x, y = move
    zero_x, zero_y = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0][0]
    new_state[zero_x][zero_y], new_state[x][y] = new_state[x][y], new_state[zero_x][zero_y]
    return new_state

def get_lowest_cost_node(open_list):
    lowest_cost_node = open_list[0]
    lowest_cost_index = 0
    for i, node in enumerate(open_list[1:], start=1):
        if node.cost < lowest_cost_node.cost:
            lowest_cost_node = node
            lowest_cost_index = i
    return lowest_cost_index

def solve_puzzle(initial_state):
    visited = set()
    open_list = [Node(initial_state)]
    
    while open_list:
        current_index = get_lowest_cost_node(open_list)
        current_node = open_list.pop(current_index)
        current_node.statistics = f"f(n) [{current_node.cost}] = g(n) [{current_node.depth}] + h(n) [{misplaced_tiles(current_node.state)}]"
        
        if current_node.state == goal_state:
            moves = []
            while current_node.parent:
                moves.append({"move":current_node.move,
                              "stats": current_node.statistics})
                current_node = current_node.parent
            moves.reverse()
            return moves
        
        visited.add(tuple(map(tuple, current_node.state)))
        moves = possible_moves(current_node.state)
        
        for move in moves:
            new_state = get_new_state(current_node.state, move)
            if tuple(map(tuple, new_state)) not in visited:
                new_node = Node(new_state, current_node, move, current_node.depth + 1)
                open_list.append(new_node)
    
    return None

def print_puzzle(state):
    for row in state:
        print(" ".join(str(cell) if cell != 0 else ' ' for cell in row))
    print()

if __name__ == '__main__':

    # if input("Would you like to randomly generate the initial state, or enter it? (R for Random/I for Input): ") in "Ii":
    #     print("Enter the initial state for the 8-puzzle (use digits 0-8, space separated, row by row):")
    #     initial_state = []
    #     while len(initial_state) < 3:
    #         row = input(f"Enter row {len(initial_state) + 1}: ").strip().split()
    #         if len(row) == 3:
    #             initial_state.append([int(num) for num in row])
    #         else:
    #             print("Please enter exactly 3 digits per row.")
    # else:
    #     numbers = list(range(9))
    #     random.shuffle(numbers)
    #     initial_state = [numbers[i:i + 3] for i in range(0, 9, 3)]


    # print("Enter the goal state for the 8-puzzle (use digits 0-8, space separated, row by row):")
    # goal_state = []
    # while len(goal_state) < 3:
    #     row = input(f"Enter row {len(goal_state) + 1}: ").strip().split()
    #     if len(row) == 3:
    #         goal_state.append([int(num) for num in row])
    #     else:
    #         print("Please enter exactly 3 digits per row.")

    # Define the goal state
    goal_state = [
        [1, 2, 3],
        [8, 0, 4],
        [7, 6, 5]
    ]

    # Define the initial state
    initial_state = [
        [2, 8, 3],
        [1, 6, 4],
        [7, 0, 5]
    ]

    # print("Looking for solutions...")

    solution = solve_puzzle(initial_state)
    os.system("cls")

    print("""\nSolution found:
g(n) = cost so far to reach
h(n) = estimated cost to goal from n
f(n) = estimated total cost of path through n to goal

A∗ search uses an admissible heuristic
i.e., h(n) ≤ h∗(n) where h∗(n) is the true cost from n.
(Also require h(n) ≥ 0, so h(G) = 0 for any goal G.)

""")


    if solution:
        current_state = initial_state
        print("Initial State:")
        print_puzzle(current_state)

        for move in solution:
            current_state = get_new_state(current_state, move["move"])
            print("Move:", move["move"])
            print("Statistics:", move["stats"])
            print_puzzle(current_state)
        print("\n\nFinal Cost:",move["stats"])
    else:
        print("No solution found.")