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
        self.depth = 2
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
            self.children = []
            #Tree level of node (even = max node, odd = min node)
            self.level = 0
            self.end = None
            self.endscore=0

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        start_time = time.time()
        self.timer = start_time
        self.turn +=1
        flag = False
        #Create root node
        if self.root is None:
            self.root = self.Node(my_pos)
            self.board_size = len(chess_board)
            self.max_step = max_step
        elif len(self.root.children)!=0:
            bound = 0
            #check adv pos, find new boundary, 
            for i,b in enumerate(self.chess_board[adv_pos[0]][adv_pos[1]]):
                if (b==False) and (chess_board[adv_pos[0]][adv_pos[1]][i]==True):
                    bound = i
                    break
            # Update root node to the current position
            for c in self.root.children:
                if np.array_equal(adv_pos, c.pos) and bound==c.boundary:
                    flag = True
                    self.root=c
                    self.root.pos=my_pos
                    self.root.level=0
                    self.root.boundary=None
                    break
            # Sanity check: in case state not found in previously computed states, create new root node.
            if not flag:
                self.root = self.Node(my_pos)
        else:
            self.root = self.Node(my_pos)
                    
                    
            
        self.chess_board = deepcopy(chess_board)
            
        pos, b = self.minimax_decision(adv_pos)    

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        return pos, b
    
    def check_valid_step(self, board, start_pos, end_pos, adv_pos, barrier_dir):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

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
            board (numpy.ndarray): _description_
            pos (list of int): _description_
            pos_adv (list of int): _description_
            max_step (int): _description_
            prev_dir (int): _description_
            node (Node): _description_
            visited (dict, optional): _description_. Defaults to {}.

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
        visited.add(tuple(pos))    
        return node
    
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
        for c in self.root.children:
            #Copy board
            boardc = self.get_copy_board(self.chess_board, c.pos, c.boundary)
            c.level=1
            val, win= self.minimax_value(c, boardc, c.pos, pos_adv, alpha, float('inf'))
            # If step results in an immediate win, return step
            if win:
                return c.pos, c.boundary
            if val>alpha:
                alpha = val
                max_node = c
            if time.time()-self.timer>1.97:
                self.update_board_root(max_node)
                return max_node.pos, max_node.boundary
                
        #Update root and board
        self.update_board_root(max_node)
        return max_node.pos, max_node.boundary
    
    def evaluation(self, node, board, pos, pos_adv):
        
        # Get children first, to determine stuff like how many moves they can make
        boardc = self.get_copy_board(board, node.pos, node.boundary)
        
        # #Feature 1: Number of moves I can make from this position
        self.get_children(boardc, pos_adv, pos, self.max_step, -1, node)
        if node.level%2!=0:
            return -len(node.children)
        return len(node.children)

    
    def minimax_value(self, node, board, pos, pos_adv, alpha, beta):
        #check for end of game, if the move leads to a win, immediately return and make that move
        if node.end is None and not(self.board_size>6 and self.turn==1):
            end, score1, score2 = self.check_endgame(board, pos, pos_adv)
            if end:
                node.end=True
                node.endscore = score1-score2 if node.level%2!=0 else score2-score1
            else:
                node.end = False
        if node.end:
            if node.level==1 and node.endscore>0: 
                return node.endscore, True
            #If move gives opponent an opening for a win in following turn, return immediately
            if node.level<3 and node.endscore<0:
                return -(self.board_size*self.board_size), False
            return (self.board_size*self.board_size)/2+node.endscore, False
        
        if time.time()-self.timer>1.97:
            return 0, False
        
        #utility computed using evaluation function if depth reached
        if self.depth==node.level:
            return self.evaluation(node, board, pos, pos_adv), False
            # return 0, False
            # return sum(bool(x) for x in board[pos[0]][pos[1]]), False
        
        #get children
        if len(node.children)==0:
            self.get_children(board, pos_adv, node.pos, self.max_step, -1, node)
        print("LEVEL" + str(node.level+1))
        #get utility 
        m_ind = None
        for i,n in enumerate(node.children):
            n.level = node.level+1
            boardc = self.get_copy_board(board, n.pos, n.boundary)
            val, win= self.minimax_value(n, boardc, n.pos, node.pos, alpha, beta)
            if node.level%2!=0:
                beta = val if val<beta else beta
                m_ind = i if val<beta else m_ind
            else:
                alpha = val if val>alpha else alpha
                m_ind = i if val>alpha else m_ind
            if alpha>= beta:
                # Place child that caused pruning at front of list.
                if i!=0:
                    node.children.insert(0, node.children.pop(i))
                break
        return alpha if node.level%2==0 else beta, False
        
   
  
