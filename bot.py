import chess
import random
import copy
import json

# =====================
# ChessBot Class
# =====================
class ChessBot:
    def __init__(self, depth=3):
        # Search depth
        self.depth = depth

        # Tunable piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Piece-square tables (simplified, add more for full bot)
        self.position_scores = {
            chess.PAWN: [0,0,0,0,0,0,0,0,
                         50,50,50,50,50,50,50,50,
                         10,10,20,30,30,20,10,10,
                         5,5,10,25,25,10,5,5,
                         0,0,0,20,20,0,0,0,
                         5,-5,-10,0,0,-10,-5,5,
                         5,10,10,-20,-20,10,10,5,
                         0,0,0,0,0,0,0,0],
            chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50,
                           -40,-20,0,0,0,0,-20,-40,
                           -30,0,10,15,15,10,0,-30,
                           -30,5,15,20,20,15,5,-30,
                           -30,0,15,20,20,15,0,-30,
                           -30,5,10,15,15,10,5,-30,
                           -40,-20,0,5,5,0,-20,-40,
                           -50,-40,-30,-30,-30,-30,-40,-50],
            chess.BISHOP: [
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 10, 10, 5, 0, -10,
                -10, 5, 5, 10, 10, 5, 5, -10,
                -10, 0, 10, 10, 10, 10, 0, -10,
                -10, 10, 10, 10, 10, 10, 10, -10,
                -10, 5, 0, 0, 0, 0, 5, -10,
                -20, -10, -10, -10, -10, -10, -10, -20
            ],
            chess.ROOK: [
                0, 0, 0, 5, 5, 0, 0, 0,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                5, 10, 10, 10, 10, 10, 10, 5,
                0, 0, 0, 0, 0, 0, 0, 0
            ],
            chess.QUEEN: [
                -20, -10, -10, -5, -5, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 5, 5, 5, 0, -10,
                -5, 0, 5, 5, 5, 5, 0, -5,
                0, 0, 5, 5, 5, 5, 0, -5,
                -10, 5, 5, 5, 5, 5, 0, -10,
                -10, 0, 5, 0, 0, 0, 0, -10,
                -20, -10, -10, -5, -5, -10, -10, -20
            ],
            chess.KING: [
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -20, -30, -30, -40, -40, -30, -30, -20,
                -10, -20, -20, -20, -20, -20, -20, -10,
                20, 20, 0, 0, 0, 0, 20, 20,
                20, 30, 10, 0, 0, 10, 30, 20
            ]
        }

        # Opening book (learned via self-play)
        self.opening_book = {}

        # Transposition table for alpha-beta pruning
        self.transposition_table = {}

    # ---------------------
    # Get move
    # ---------------------
    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Check opening book first
        book_move = self.get_book_move(board)
        if book_move and book_move in legal_moves:
            return book_move

        # Minimax search
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in legal_moves:
            board.push(move)
            value = -self.minimax(board, self.depth - 1, -beta, -alpha, False)
            board.pop()

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, value)

        return best_move

    # ---------------------
    # Minimax with alpha-beta
    # ---------------------
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        board_hash = self.get_board_hash(board)
        if board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            return self.transposition_table[board_hash]['value']

        if depth == 0 or board.is_game_over():
            value = self.evaluate_position(board)
            self.transposition_table[board_hash] = {'value': value, 'depth': depth}
            return value

        legal_moves = list(board.legal_moves)

        if maximizing_player:
            value = float('-inf')
            for move in legal_moves:
                board.push(move)
                value = max(value, self.minimax(board, depth - 1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float('inf')
            for move in legal_moves:
                board.push(move)
                value = min(value, self.minimax(board, depth - 1, alpha, beta, True))
                board.pop()
                beta = min(beta, value)
                if alpha >= beta:
                    break

        self.transposition_table[board_hash] = {'value': value, 'depth': depth}
        return value

    # ---------------------
    # Evaluation
    # ---------------------
    def evaluate_position(self, board):
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        material_score = 0
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.piece_type in self.position_scores:
                    idx = square if piece.color == chess.WHITE else 63 - square
                    value += self.position_scores[piece.piece_type][idx] / 10
                material_score += value if piece.color == chess.WHITE else -value

        return material_score if board.turn == chess.WHITE else -material_score

    # ---------------------
    # Opening Book
    # ---------------------
    def get_book_move(self, board):
        fen = board.board_fen() + ' ' + ('w' if board.turn == chess.WHITE else 'b')
        if fen in self.opening_book:
            moves = self.opening_book[fen]
            total_weight = sum(weight for _, weight in moves)
            if total_weight <= 0:
                return None
            r = random.random() * total_weight
            cumulative = 0
            for move, weight in moves:
                cumulative += weight
                if r <= cumulative:
                    return move
        return None

    def update_opening_book(self, game_result, moves, color):
        if (game_result == 1 and color == chess.WHITE) or (game_result == -1 and color == chess.BLACK) or game_result == 0:
            board = chess.Board()
            for move in moves[:15]:
                fen = board.board_fen() + ' ' + ('w' if board.turn == chess.WHITE else 'b')
                if fen not in self.opening_book:
                    self.opening_book[fen] = []
                move_found = False
                for i, (book_move, weight) in enumerate(self.opening_book[fen]):
                    if book_move == move:
                        self.opening_book[fen][i] = (book_move, weight + (5 if game_result != 0 else 2))
                        move_found = True
                        break
                if not move_found:
                    self.opening_book[fen].append((move, 10 if game_result != 0 else 5))
                board.push(move)

    def get_board_hash(self, board):
        return board.board_fen()

# =====================
# Self-Play Trainer
# =====================
class SelfPlayTrainer:
    def __init__(self, bot_constructor, games_per_iteration=50, iterations=10):
        self.bot_constructor = bot_constructor
        self.games_per_iteration = games_per_iteration
        self.iterations = iterations
        self.best_bot = bot_constructor()
        self.performance_history = []

    def train(self):
        for iteration in range(self.iterations):
            print(f"Starting iteration {iteration+1}/{self.iterations}")
            challenger = self.create_challenger()
            results = self.play_match(self.best_bot, challenger)
            analysis = self.analyze_results(results)
            print(f"Iteration results: {analysis}")
            self.update_best_bot(challenger, analysis)
            self.performance_history.append({"iteration": iteration+1, "metrics": analysis})
        return self.best_bot

    def create_challenger(self):
        challenger = self.bot_constructor()
        if hasattr(challenger, 'piece_values'):
            for piece in challenger.piece_values:
                challenger.piece_values[piece] *= random.uniform(0.95, 1.05)
        if hasattr(challenger, 'position_scores'):
            for piece in challenger.position_scores:
                for i in range(len(challenger.position_scores[piece])):
                    challenger.position_scores[piece][i] *= random.uniform(0.95, 1.05)
        return challenger

    def play_match(self, bot1, bot2):
        results = []
        for game_num in range(self.games_per_iteration):
            board = chess.Board()
            moves = []
            result_code = self.play_game(board, bot1, bot2, moves, game_num)
            results.append({"game_num": game_num, "winner": result_code, "moves": moves})
        return results

    def play_game(self, board, bot1, bot2, moves, game_num):
        move_count = 0
        position_history = {}
        while not board.is_game_over():
            key = board.board_fen()
            position_history[key] = position_history.get(key, 0) + 1
            if position_history[key] >= 3 or board.halfmove_clock >= 100:
                return 'draw'
            current_bot = bot1 if board.turn == chess.WHITE else bot2
            move = current_bot.get_move(board)
            if move is None:
                break
            moves.append(move)
            board.push(move)
            move_count += 1
            if move_count > 200:
                return 'draw'
        if board.is_checkmate():
            return 'bot1' if board.turn == chess.BLACK else 'bot2'
        return 'draw'

    def analyze_results(self, results):
        bot1_wins = sum(1 for r in results if r['winner']=='bot1')
        bot2_wins = sum(1 for r in results if r['winner']=='bot2')
        draws = sum(1 for r in results if r['winner']=='draw')
        total = len(results)
        return {
            'bot1_wins': bot1_wins, 'bot2_wins': bot2_wins, 'draws': draws,
            'bot1_win_rate': bot1_wins/total, 'bot2_win_rate': bot2_wins/total, 'draw_rate': draws/total
        }

    def update_best_bot(self, challenger, analysis):
        if analysis['bot2_win_rate'] > analysis['bot1_win_rate']:
            self.best_bot.piece_values = copy.deepcopy(challenger.piece_values)
            self.best_bot.position_scores = copy.deepcopy(challenger.position_scores)

    def save_best_bot(self, filepath):
        params = {'piece_values': self.best_bot.piece_values, 'position_scores': self.best_bot.position_scores}
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)

    def save_training_history(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
