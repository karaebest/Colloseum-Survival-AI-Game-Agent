import numpy as np
import time
import student_agent
opposites = {0: 2, 1: 3, 2: 0, 3: 1}
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
def make_board(size, bounds):
    
    chess_board = np.zeros((size, size, 4), dtype=bool)
    chess_board[0, :, 0] = True
    chess_board[:, 0, 3] = True
    chess_board[-1, :, 2] = True
    chess_board[:, -1, 1] = True
    
    for b in bounds:
        chess_board[b[0]][b[1]][b[2]]=True
        chess_board[b[0]+moves[b[2]][0]][b[1]+moves[b[2]][1]][opposites[b[2]]]=True
    return chess_board

def update_board(board, pos, b):
    board[pos[0]][pos[1]][b]=True
    board[pos[0]+moves[b][0]][pos[1]+moves[b][1]][opposites[b]]=True

p0_pos = [0,1]
p1_pos = [3,2]
player_names = ["p0","p1"]

#bounds = [[0,1,1],[0,1,2],[0,2,2],[1,1,2],[1,2,2],[2,1,1],[2,1,2],[2,2,1]]
bounds = [[0,2,1],[3,0,1]]
board = make_board(4, bounds)

p0 = student_agent.StudentAgent()
new_pos, b = p0.step(board, p0_pos, p1_pos, 2)
print(str(new_pos)+" "+str(b))


update_board(board, new_pos, b)
p1_pos = [1,2]
b1 = 3
update_board(board, p1_pos, b1)
new_pos, b = p0.step(board, new_pos, p1_pos,2)
print(str(new_pos)+" "+str(b))