import numpy as np
from copy import deepcopy 
import time

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
depth = 3
root = None
t = time.time()

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
        if board[pos[0]][pos[1]][dir]:
            continue
        #Create child node and add to list of visited positions
        c = Node(pos)
        c.level=node.level+1
        c.boundary=dir
        node.children.append(c)
        #Ensure to not move back to previous position
        if prev_dir==-1 or (prev_dir!= -1 and dir != opposites[prev_dir]):
            next_pos = np.array(pos)+np.array(moves[dir])
            #Check if able to move to next position (still able to take step and adversary not in that position)
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
        vals[i], win= minimax_value(c, boardc, my_pos, pos_adv)
        # if time.time()-t>1.89:
        #     c_max = root.children[vals[0:i+1].argmax()]
        #     return c_max.pos, c_max.boundary
        # If step results in an immediate win, return step
        if win:
            return c.pos, c.boundary         
    c_max = root.children[vals.argmax()]
    
    return c_max.pos, c_max.boundary

def evaluation(node, board, pos, pos_adv):
    # Get children first, to determine stuff like how many moves they can make
    boardc = deepcopy(board)
    #Set boundaries in child state
    boardc[node.pos[0]][node.pos[1]][node.boundary]=True
    boardc[node.pos[0]+moves[node.boundary][0]][node.pos[1]+moves[node.boundary][1]][opposites[node.boundary]]=True
    
    #Feature 1: Number of moves I can make from this position (regardless of move opponent makes)
    nodec = deepcopy(node)
    get_children(boardc, pos, pos_adv, max_step, -1, nodec)
    feat1 = len(nodec.children)
    #Feature 2: Number of moves opponent can make
    get_children(boardc, pos_adv, pos, max_step, -1, node)
    feat2 = len(node.children)
    
    #Normalize by something so the winning score doesn't overpower this

    #Feature 2: Zone expansion
        # count reachable spaces
    print("score for pos:"+str(node.pos)+" and boundary: "+ str(node.boundary) + " At level : "+ str(node.level))
    score = feat1-feat2
    print(score)
    
    return score

# Returns tuple: utility, immediate win
def minimax_value(node, board, pos, pos_adv):
    #check for end of game, if the move leads to a win, immediately return and make that move
    end, score1, score2 = check_endgame(board, pos, pos_adv)
    if end:
        #If next move leads to immediate win, return immediately
        if node.level==1: 
            if score1>score2:
                print("instant win")
                return score1, True
            else:
                print("instant loss: "+str(node.pos))
                # Ensure score will be lower than any other possible score
                return -(len(board)*len(board)), False
        if node.level%2!=0:
            print("endgame at min node: "+ str(node.pos)+" "+str(node.boundary)+" level:"+str(node.level)+ " score: "+ str(score1-score2))
            return score1-score2, False
        else: 
            print("endgame at max node: "+ str(pos_adv)+" level:"+str(node.level)+ " score:"+ str(score1-score2))
            return score2-score1, False
            

    #check if depth has been reached
    if depth==node.level:
        return evaluation(node, board, pos, pos_adv, False), False
    
    #get children
    if len(node.children)==0:
        get_children(board, pos_adv, pos, max_step, -1, node)
        
    #get utility
    vals=np.zeros(len(node.children))
    for i,n in enumerate(node.children):
        boardc = deepcopy(board)
        #Set boundaries in child state
        boardc[n.pos[0]][n.pos[1]][n.boundary]=True
        boardc[n.pos[0]+moves[n.boundary][0]][n.pos[1]+moves[n.boundary][1]][opposites[n.boundary]]=True
        vals[i],win=minimax_value(n, boardc, pos_adv, pos)
    if node.level%2==0:
        m = np.max(vals)
        print("max node util: "+ str(m)+ "at level: "+ str(node.level))
        
        return np.max(vals), False
    else:
        m = np.min(vals)
        print("min node util: "+ str(m) + "at level: "+ str(node.level))
        return np.min(vals), False

######################## MAKE BOARD###############
p0_pos = [2,2]
p1_pos = [1,0]
player_names = ["p0","p1"]

#bounds = [[0,1,1],[0,1,2],[0,2,2],[1,1,2],[1,2,2],[2,1,1],[2,1,2],[2,2,1]]
bounds = [[1,0,2],[1,1,2],[1,2,2]]
board = make_board(4, bounds)

root = Node(p0_pos)

########### GET CHILDREN TEST################
# root = get_children(board, p0_pos, p1_pos, 2, -1, root)

# for c in root.children:
#     print("=====")
#     print(c.pos)
#     print(c.boundary)
#     assert(c.level==1)
    
######### CHECK ENDGAME TEST############

# print(check_endgame(board, p0_pos, p1_pos))

# board[3][0][1]=True
# board[3][1][3]=True

# print(check_endgame(board, p0_pos, p1_pos))


######### MINIMAX TEST#############

pos, boundary = minimax_decision(root, board, p0_pos, p1_pos)
print(pos)
print(boundary)
turn = time.time()
print(turn-t)

