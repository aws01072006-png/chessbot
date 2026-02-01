import chess
from board import ChessBoard
import random


class ChessBot:
    def __init__(self):
        pass

    def get_move(self, board: ChessBoard):
        """
        Given the current board state, returns the chosen move.
        This is the main function students will implement.
        """
        legal_moves = board.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)
        return None
