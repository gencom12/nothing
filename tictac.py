import math
from graphviz import Digraph

class TicTacToe:
    PLAYER_X = 1
    PLAYER_O = -1
    EMPTY = 0

    def __init__(self):
        self.board = [[self.EMPTY, self.EMPTY, self.EMPTY],
                      [self.EMPTY, self.EMPTY, self.EMPTY],
                      [self.EMPTY, self.EMPTY, self.EMPTY]]

    def print_board(self):
        for row in self.board:
            print(" ".join(["X" if cell == self.PLAYER_X else "O" if cell == self.PLAYER_O else "-" for cell in row]))

    def evaluate(self):
        winner = self.check_winner()
        if winner == self.PLAYER_X:
            return 1
        elif winner == self.PLAYER_O:
            return -1
        elif self.is_board_full():
            return 0
        else:
            return None

    def is_board_full(self):
        return all(all(cell != self.EMPTY for cell in row) for row in self.board)

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != self.EMPTY:
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != self.EMPTY:
                return self.board[0][i]

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != self.EMPTY:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != self.EMPTY:
            return self.board[0][2]

        return None

    def get_available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == self.EMPTY]

    def minimax_alpha_beta(self, depth, alpha, beta, maximizing_player, graph):
        result = self.evaluate()
        if result is not None:
            return result

        if maximizing_player:
            max_eval = -math.inf
            for move in self.get_available_moves():
                i, j = move
                self.board[i][j] = self.PLAYER_X

                # Create a node in the graph
                node_name = f"{i}_{j}"
                graph.node(node_name, label=f"{i},{j}\n{alpha},{beta}")

                eval = self.minimax_alpha_beta(depth - 1, alpha, beta, False, graph)

                # Add an edge to the graph
                graph.edge(node_name, f"{i}_{j}_eval", label=f"{eval}")

                self.board[i][j] = self.EMPTY
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    graph.node(f"{i}_{j}_eval", label="Pruned", shape="box")
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in self.get_available_moves():
                i, j = move
                self.board[i][j] = self.PLAYER_O

                # Create a node in the graph
                node_name = f"{i}_{j}"
                graph.node(node_name, label=f"{i},{j}\n{alpha},{beta}")

                eval = self.minimax_alpha_beta(depth - 1, alpha, beta, True, graph)

                # Add an edge to the graph
                graph.edge(node_name, f"{i}_{j}_eval", label=f"{eval}")

                self.board[i][j] = self.EMPTY
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    graph.node(f"{i}_{j}_eval", label="Pruned", shape="box")
                    break
            return min_eval

    def find_best_move(self, graph):
        best_val = -math.inf
        best_move = None

        for move in self.get_available_moves():
            i, j = move
            self.board[i][j] = self.PLAYER_X

            # Create a node in the graph
            node_name = f"{i}_{j}"
            graph.node(node_name, label=f"{i},{j}")

            move_val = self.minimax_alpha_beta(0, -math.inf, math.inf, False, graph)

            # Add an edge to the graph
            graph.edge(node_name, f"{i}_{j}_eval", label=f"{move_val}")

            self.board[i][j] = self.EMPTY

            if move_val > best_val:
                best_move = move
                best_val = move_val

        return best_move

# Example usage:
game = TicTacToe()
game.print_board()

graph = Digraph('tictac')
graph.attr(size='8,8')

while not game.is_board_full() and game.check_winner() is None:
    x, y = map(int, input("Enter your move (row and column, separated by space): ").split())
    if game.board[x][y] == game.EMPTY:
        game.board[x][y] = game.PLAYER_O
    else:
        print("Invalid move! Try again.")
        continue

    game.print_board()

    if game.is_board_full() or game.check_winner() is not None:
        break

    print("Thinking...")
    best_move = game.find_best_move(graph)
    game.board[best_move[0]][best_move[1]] = game.PLAYER_X

    game.print_board()

winner = game.check_winner()
if winner == game.PLAYER_X:
    print("You lose! The computer wins.")
elif winner == game.PLAYER_O:
    print("Congratulations! You win.")
else:
    print("It's a tie!")

graph.render('tic_tac_toe_graph', format='png', cleanup=True)
