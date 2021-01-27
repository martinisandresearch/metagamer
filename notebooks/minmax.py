import numpy as np
from matplotlib import pyplot as plt
# board: np.array [3,3]

def b2v(board):
    """convert board from 2d representation into 1 d representation"""
    return board.reshape(-1)

def v2b(v: np.array):
    """Convert board from vector into 2d representation"""
    return v.reshape(3,3)

def prop_mvs(v):
    """proposed moves, using the 1d represntation"""
    return np.where(v==0)[0]

eg = []
eg.append(np.identity(3))
eg.append(np.fliplr(eg[-1]))
eg.append(np.array([[1,1,1],[0,0,0],[0,0,0]]))
eg.append(np.roll(eg[-1],1,axis=0))
eg.append(np.roll(eg[-1],1,axis=0))
eg.append(eg[-3].T)
eg.append(eg[-3].T)
eg.append(eg[-3].T)
    
def fin(v):
    """is the game over?"""
    score = np.dot(v,eg)
    ind = np.argmax(np.abs(score))
    
    if np.abs(score[ind])!=3:
        return 0
    else: 
        return np.sign(score[ind])
    

def gm(board, player='x'):
    """given the current board give the best move using min max trees """
    sgn = 1
    if player != 'x':
        sgn = -1

    v = b2v(board)
    mvs = prop_mvs(v)
    opts = []
    bds = []

    for i in mvs:
        new_board = np.copy(v)
        new_board[i] = sgn
        bds.append(new_board)
    
    for i in range(len(bds)):
        tree = []
        outs = []

        tree.append([bds[i]])

        while len(tree) != 0:
            while len(tree[-1]) != 0:
                out = fin(tree[-1][-1])
                mvs = prop_mvs(tree[-1][-1])
                if np.abs(out)==1:
                    outs.append(out)
                    tree[-1].pop(-1)            
                elif len(mvs) == 0:
                    outs.append(out)
                    tree[-1].pop(-1)
                else:
                    next = []
                    for i in tree:
                        pass

                # Expand
            tree.append()
            print(mvs)
        # Go up one layer:
        tree.pop(-1)
        sgn = sgn * -1  
    


board = np.sign(np.random.randn(3,3)).astype(int)+1
print(gm(board))

              