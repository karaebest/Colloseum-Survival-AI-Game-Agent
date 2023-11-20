import numpy as np
import time

# def check_endgame(board_size, chess_board, moves, p0_pos, p1_pos, player_names):
#     """
#     Check if the game ends and compute the current score of the agents.

#     Returns
#     -------
#     is_endgame : bool
#         Whether the game ends.
#     player_1_score : int
#         The score of player 1.
#     player_2_score : int
#         The score of player 2.
#     """
#     # Union-Find
#     father = dict()
#     for r in range(board_size):
#         for c in range(board_size):
#             father[(r, c)] = (r, c)

#     def find(pos):
#         if father[pos] != pos:
#             father[pos] = find(father[pos])
#         return father[pos]

#     def union(pos1, pos2):
#         father[pos1] = pos2

#     for r in range(board_size):
#         for c in range(board_size):
#             for dir, move in enumerate(
#                 moves[1:3]
#             ):  # Only check down and right
#                 if chess_board[r, c, dir + 1]:
#                     continue
#                 pos_a = find((r, c))
#                 pos_b = find((r + move[0], c + move[1]))
#                 if pos_a != pos_b:
#                     union(pos_a, pos_b)

#     for r in range(board_size):
#         for c in range(board_size):
#             find((r, c))
#     p0_r = find(tuple(p0_pos))
#     p1_r = find(tuple(p1_pos))
#     p0_score = list(father.values()).count(p0_r)
#     p1_score = list(father.values()).count(p1_r)
#     print(father)
#     if p0_r == p1_r:
#         return False, p0_score, p1_score
#     player_win = None
#     win_blocks = -1
#     if p0_score > p1_score:
#         player_win = 0
#         win_blocks = p0_score
#     elif p0_score < p1_score:
#         player_win = 1
#         win_blocks = p1_score
#     else:
#         player_win = -1  # Tie
#     if player_win >= 0:
#         print(
#             f"Game ends! Player {player_names[player_win]} wins having control over {win_blocks} blocks!"
#         )
#     else:
#         print("Game ends! It is a Tie!")
#     return True, p0_score, p1_score


board_size = 4
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
p0_pos = [2,0]
p1_pos = [0,3]
player_names = ["p0","p1"]
chess_board = np.zeros((board_size, board_size, 4), dtype=bool)
chess_board[0, :, 0] = True
chess_board[:, 0, 3] = True
chess_board[-1, :, 2] = True
chess_board[:, -1, 1] = True

chess_board[0:3, 1, 1] = True
chess_board[1:3, 2, 1] = True
chess_board[0, 1:3, 2] = True
chess_board[1, 1:3, 2] = True
chess_board[2, 1, 2] = True

# t = time.time()
# for i in range(100):
#     check_endgame(board_size, chess_board, moves, p0_pos, p1_pos, player_names)
# print(time.time()-t)

check = {-1: [[0,1,2,3],[0,1,2,3]], 0: [[0,1,3],[0,1,3]], 1: [[1,2], [1]], 2: [[1,2,3],[1,2,3]], 3: [[2,3],[3]]}
# print(check[1][1])


moves = np.array(((-1, 0), (0, 1), (1, 0), (0, -1)))
# pos = np.array([0,1])
# print(pos+moves[0])

def get_moves(board, pos, pos_adv, max, moves1, prev_dir):
    #maybe add check for if im going to box myself in
    
    for dir in check[prev_dir][0]:
        if board[pos[0]][pos[1]][dir] == False: #boundaries in current position
            moves1.append([pos[0],pos[1],dir])
            next_pos = np.array(pos)+np.array(moves[dir])
            if max!=0 and not np.array_equal(pos_adv, next_pos):
                moves1 = get_moves(board,next_pos,pos_adv,max-1,moves1,dir)

    return moves1

print(get_moves(chess_board, p0_pos, p1_pos, 2, [], -1))