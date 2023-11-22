import numpy as np
from copy import deepcopy 

        # self.dir_map = {
        #     "u": 0,
        #     "r": 1,
        #     "d": 2,
        #     "l": 3,
        # }

opposites = {0: 2, 1: 3, 2: 0, 3: 1}
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
check = {-1: [[0,1,2,3],[0,1,2,3]], 0: [[0,1,3],[0,1,3]], 1: [[1,2], [1]], 2: [[1,2,3],[1,2,3]], 3: [[2,3],[3]]}

max_step = 2
depth = 2

class Node:
    def __init__(self, pos):
        #Boundary placed to reach the state represented by this node
        self.pos = np.array(pos)
        self.boundary = None
        self.children = []
        #Tree level of node (even = max node, odd = min node)
        self.level = 0
        
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

def check_endgame(board, p0_pos, p1_pos):
    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """
    board_size = len(board)
    # Union-Find
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                moves[1:3]
            ):  # Only check down and right
                if board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))
    p0_r = find(tuple(p0_pos))
    p1_r = find(tuple(p1_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score, p1_score
    return True, p0_score, p1_score

def get_children(board, pos, pos_adv, max_step, prev_dir, node, visited={}): 
    if prev_dir==-1:
        visited = {tuple(pos)}
    #Check all boundaries in current position except direction we came from
    for dir in range(4):
        #Check if possible to set barrier
        if (prev_dir!=-1 and dir==opposites[prev_dir]) or board[pos[0]][pos[1]][dir] == True:
            continue
        #Create child node and add to list of visited positions
        c = Node(pos)
        c.level=node.level+1
        c.boundary=dir
        node.children.append(c)
         
        #check if possible to move in that direction
        next_pos = np.array(pos)+np.array(moves[dir])
        if (not tuple(next_pos) in visited) and max_step!=0 and not np.array_equal(pos_adv, next_pos):
            visited.add(tuple(next_pos))
            get_children(board, next_pos, pos_adv, max_step-1, dir, node, visited)
    return node

def minimax_decision(root, board, my_pos, pos_adv):
    get_children(board, my_pos, pos_adv, max_step, -1, root)
    vals = np.zeros(len(root.children))
    for i,c in enumerate(root.children):
        #Copy board
        boardc = deepcopy(board)
        #Set boundaries in child state
        boardc[c.pos[0]][c.pos[1]][c.boundary]=True
        boardc[c.pos[0]+moves[c.boundary][0]][c.pos[1]+moves[c.boundary][1]][opposites[c.boundary]]=True
        vals[i] = minimax_value(c, boardc, my_pos, pos_adv)
    c_max = root.children[vals.argmax()]
    step=np.array([c_max.pos[0], c_max.pos[1], c_max.boundary])
    return step

def evaluation(node, board):
    return 0

def minimax_value(node, board, pos, pos_adv):
    #check for end of game
    end, score1, score2 = check_endgame(board, pos, pos_adv)
    print("==============")
    print(end)
    print(node.level)
    print(node.pos)
    print(node.boundary)
    if end:
        if node.level%2!=0:
            print(score1)
            return score1
        print(score2)
        return score2
    #check if depth has been reached
    if depth==node.level:
        return evaluation(node, board)
    #get children
    if len(node.children)==0:
        #Copy board
        boardc = deepcopy(board)
        #Set boundaries in child state
        boardc[node.pos[0]][node.pos[1]][node.boundary]=True
        boardc[node.pos[0]+moves[node.boundary][0]][node.pos[1]+moves[node.boundary][1]][opposites[node.boundary]]=True
        get_children(boardc, pos_adv, pos, max_step, -1, node)
    #get utility
    vals=np.zeros(len(node.children))
    for i,n in enumerate(node.children):
        vals[i]=minimax_value(n, boardc, pos_adv, pos)
    if node.level%2==0:
        return np.max(vals)
    else:
        return np.min(vals)

######################## MAKE BOARD###############
p0_pos = [1,2]
p1_pos = [0,3]
player_names = ["p0","p1"]

bounds = [[0,1,1],[0,1,2],[0,2,2],[1,1,2],[1,2,2],[2,1,1],[2,1,2],[2,2,1]]
board = make_board(4, bounds)

root = Node(p1_pos)

########### GET CHILDREN TEST################
root = get_children(board, p0_pos, p1_pos, 2, -1, root)

for c in root.children:
    print("=====")
    print(c.pos)
    print(c.boundary)
    
######### CHECK ENDGAME TEST############

# print(check_endgame(board, p0_pos, p1_pos))

# board[3][0][1]=True
# board[3][1][3]=True

# print(check_endgame(board, p0_pos, p1_pos))


######### MINIMAX TEST#############

# minimax_decision(root, board, p0_pos, p1_pos)