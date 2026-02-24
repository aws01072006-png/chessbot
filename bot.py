import chess
from board import ChessBoard
import random


class ChessBot:
    def __init__(self):
        pass

    def get_move(self, board: ChessBoard):
        legal_moves = board.get_legal_moves()
        board_state = board.get_board_state()
        is_over = board.is_game_over()
        if legal_moves:
            return random.choice(legal_moves)
        return None
