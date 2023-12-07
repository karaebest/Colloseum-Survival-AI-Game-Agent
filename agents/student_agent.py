# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
    
        

@register_agent("student_agent")
class StudentAgent(Agent):
# class StudentAgent():
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        
        
        self.chess_board = None
        self.depth = 1
        self.max_step = 0
        self.root = None
        self.board_size = 0
        self.turn = 0
        self.timer = 0
    class Node:
        def __init__(self, pos):
            #Boundary placed to reach the state represented by this node
            self.pos = pos
            self.boundary = None
            # self.children = {}
            self.children=[]
            #Tree level of node (even = max node, odd = min node)
            self.level = 0
            self.end = None
            self.utility=0
            self.remove = False

    def step(self, chess_board, my_pos, adv_pos, max_step):
            
        self.timer = time.time()
        self.turn +=1
        root_found = False
        #Create root node
        if self.root is None:
            self.root = self.Node(my_pos)
            self.board_size = len(chess_board)
            self.max_step = max_step
        elif len(self.root.children)!=0 and self.depth>2:
            bound = 0
            #check adv pos, find new boundary, 
            for i,b in enumerate(self.chess_board[adv_pos[0]][adv_pos[1]]):
                if (b==False) and (chess_board[adv_pos[0]][adv_pos[1]][i]==True):
                    bound = i
                    break
            # Update root node to the current position
            for c in self.root.children:
                if np.array_equal(adv_pos, c.pos) and bound==c.boundary:
                    root_found = True
                    self.root=c
                    self.root.pos=my_pos
                    self.root.level=0
                    self.root.boundary=None
                    break
            # Sanity check: in case state not found in previously computed states, create new root node.
            if not root_found:
                self.root = self.Node(my_pos)
        else:
            self.root = self.Node(my_pos)
        self.depth=1
        self.chess_board = deepcopy(chess_board)  
        pos, b = self.minimax_decision(adv_pos)       
        return pos, b
    
    def check_valid_step(self, board, start_pos, end_pos, adv_pos, barrier_dir):
        """
        Check if the step the agent takes is valid (reachable and within max steps). (COPIED FROM WORLD.PY)

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is border
        r, c = end_pos
        if board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached
    
    def check_endgame(self, board, p0_pos, p1_pos):
        """
        Check if the game ends and compute the current score of the agents. (COPIED FROM WORLD.PY)

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        return True, p0_score, p1_score
    
    
    def get_children(self, board, pos, pos_adv, max_step, prev_dir, node, visited={}): 
        
        """Computes all children of given node

        Args:
            board (numpy.ndarray): board at state represented by node
            pos (list of int): position of player whose moves are being generated
            pos_adv (list of int): adversary's position
            max_step (int): max number of steps left (= self.max_step on first call, deincremented for recursive calls)
            prev_dir (int): previous direction moved in to get to pos (= -1 on first call)
            node (Node): node whose children are being generated
            visited (dict, optional): list of fully explored positions. Defaults to {}.

        """

        if prev_dir==-1:
            visited={tuple(pos)}
        #Check all boundaries in current position
        for dir in range(4):
            #Check if possible to set barrier or move in that direction
            if board[pos[0]][pos[1]][dir]:
                continue
            
            #Create child node and add to list of visited positions
            if (not tuple(pos) in visited) or prev_dir==-1:
                
                c = self.Node(pos)
                c.level=node.level+1
                c.boundary=dir
                node.children.append(c)
                
            #Ensure to not move back to previous position
            if max_step!=0 and (prev_dir==-1 or dir!=self.opposites[prev_dir]):
                #check if possible to move in that direction
                next_pos = np.array(pos)+np.array(self.moves[dir])
                if next_pos[0]!=pos_adv[0] or next_pos[1]!=pos_adv[1]:
                    self.get_children(board, next_pos, pos_adv, max_step-1, dir, node, visited)
        # Add node to list of visited nodes
        visited.add(tuple(pos))
           
        if prev_dir==-1:
            # Before returning, order children by their proximity to the adversary
            def sorting_heuristic(cnode):
                return -(abs(cnode.pos[0]-pos_adv[0])+abs(cnode.pos[1]-pos_adv[1]))
            node.children.sort(key=sorting_heuristic) 
        return
    
    def get_copy_board(self, board, pos, boundary):
        """Creates copy of board with added given boundary

        Args:
            board : numpy.ndarray
                board configuration
            pos : int
                position to add new boundary
            boundary : int
                direction of boundary to add

        Returns:
            boardc : numpy.ndarray
                deep copy of board with boundary at pos added
        """
        boardc = deepcopy(board)
        #Set boundaries in child state
        boardc[pos[0]][pos[1]][boundary]=True
        boardc[pos[0]+self.moves[boundary][0]][pos[1]+self.moves[boundary][1]][self.opposites[boundary]]=True
        return boardc
    
    def update_board_root(self, node):
        """Updates the current board configuration and root node

        Args:
            node (Node): new root node
        """
        self.chess_board[node.pos[0]][node.pos[1]][node.boundary]=True
        self.chess_board[node.pos[0]+self.moves[node.boundary][0]][node.pos[1]+self.moves[node.boundary][1]][self.opposites[node.boundary]]=True
        self.root = node
        return
    
    def minimax_decision(self, pos_adv):
        """Determines step to take using minimax logic and alpha-beta pruning

        Args:
            pos_adv (list of int): Adversary's position

        Returns:
            tuple(list of int, int): step to take
        """
        if len(self.root.children)==0:
            self.get_children(self.chess_board, self.root.pos, pos_adv, self.max_step, -1, self.root)
        max_node = None
        alpha = float('-inf')
        losses = 0
        safety_node = None
        for c in self.root.children:
            #Do not consider nodes that have been flagged to be removed
            if c.remove and losses<len(self.root.children)-1:
                if not c.end:
                    safety_node = c
                losses+=1
                continue
            #Copy board
            boardc = self.get_copy_board(self.chess_board, c.pos, c.boundary)
            c.level=1
            val, win= self.minimax_value(c, boardc, c.pos, pos_adv, alpha, float('inf'))
            # If step results in an immediate win, return step
            if win:
                return c.pos, c.boundary
            if c.remove and losses<len(self.root.children)-1:
                if not c.end:
                    safety_node = c
                losses+=1
                continue
            if val>alpha:
                alpha = val
                max_node = c
            # Check time: return if time constraint will be exceeded
            if time.time()-self.timer>1.96:
                if self.depth>2:
                    self.update_board_root(max_node)
                return max_node.pos, max_node.boundary
        if losses==len(self.root.children):
            if safety_node is not None:
                return safety_node.pos, safety_node.boundary
            else:
                return self.root.children[0].pos, self.root.children[0].boundary
        #Iterative deepening 
        if time.time()-self.timer<=0.4*self.depth and time.time()-self.timer<1.5 and self.depth<20:
            self.depth+=1
            #Sort children based on previously calculated utility
            def sorting_heuristic(n):
                return n.utility
            self.root.children.sort(key=sorting_heuristic, reverse=True)
            return self.minimax_decision(pos_adv)        
        #Update root and board if it can be used in following turn
        if self.depth>2:
            self.update_board_root(max_node)
        return max_node.pos, max_node.boundary
    
    def evaluation(self, node, board, pos, pos_adv):
        """Compute utility of node using evaluation function

        Args:
            node (Node): node
            board (numpy.ndarray): board configuration for node
            pos (list of int): current position
            pos_adv (list of int): adversary's position

        Returns:
            int: utility of node
        """
        # Feature 1: proximity to player
        feat1 = self.board_size*2-abs(pos_adv[0]-node.pos[0] + pos_adv[1]-node.pos[1])
        feat2 = 0
        # if self.turn>15 or self.board_size<9:
        # Feature 2: Number of moves I can make from this state 
        boardc = self.get_copy_board(board, node.pos, node.boundary)
        self.get_children(boardc, pos_adv, pos, self.max_step, -1, node)
        feat2 = len(node.children)/self.max_step 
        if node.level%2!=0: 
            node.utility = 2*feat1-feat2
            #Ensure adversary cannot box me in on their turn 
            if node.level==1 and sum(bool(x) for x in boardc[node.pos[0]][node.pos[1]])>2 and feat1>self.board_size+self.max_step:
                node.utility = -(self.board_size*self.board_size)    
            
        else:
            node.utility = 2*feat1+feat2 
        
        return node.utility

    
    def minimax_value(self, node, board, pos, pos_adv, alpha, beta):
        """Determines utility of node and implements alpha-beta pruning

        Args:
            node (Node): node
            board (numpy.ndarray): board configuration of node
            pos (list of int): current position
            pos_adv (list of int): adversary's position
            alpha (float): alpha value
            beta (float): beta value

        Returns:
            int: utility of node
            bool: True if immediate win (node level 1)
        """
        #check for end of game, if the move leads to a win, immediately return and make that move
        if node.end is None and not(node.level==1 and self.turn==1 and self.board_size>6):
            end, score1, score2 = self.check_endgame(board, pos, pos_adv)
            if end:
                node.end=True
                if node.level%2!=0:
                    node.utility = self.board_size*self.board_size if score1-score2>0 else -(self.board_size*self.board_size)
                else:
                    node.utility = self.board_size*self.board_size if score2-score1>0 else -(self.board_size*self.board_size)
                    
                
            else:
                node.end = False
        if node.end:
            #Flag node so that it can be removed:
            if node.utility<0 and node.level<3:
                node.remove=True
            #Return utility
            if node.level==1 and node.utility>0: 
                return node.utility, True
            return node.utility, False
        
        #Time check. 
        if time.time()-self.timer>1.97:
            return node.utility, False
        
        #Utility computed using evaluation function if depth reached
        if self.depth==node.level:
            node.utility = self.evaluation(node, board, pos, pos_adv)
            return node.utility, False
        
        #Get children
        if len(node.children)==0:
            self.get_children(board, pos_adv, node.pos, self.max_step, -1, node)
        
        #Get utility 
        m_ind = None
        for i,n in enumerate(node.children):
            n.level = node.level+1
            boardc = self.get_copy_board(board, n.pos, n.boundary)
            #Get utility of each child
            val, win= self.minimax_value(n, boardc, n.pos, node.pos, alpha, beta)            
            n.utility = val
            #Flag node to be removed if move gives opening to opponent for a win
            if node.level==1 and n.remove:
                node.remove = True
                break
            #Update alpha or beta value depending on min/max node
            if node.level%2!=0:
                beta = val if val<beta else beta
                m_ind = i if val<beta else m_ind
            else:
                alpha = val if val>alpha else alpha
                m_ind = i if val>alpha else m_ind
            # Prune node if alpha >= beta
            if alpha>= beta:
                #Killer heuristic
                node.children.insert(0,node.children.pop(i))
                break
        
        return alpha if node.level%2==0 else beta, False
        
   
  
