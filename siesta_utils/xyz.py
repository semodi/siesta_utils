import numpy as np 
import pandas as pd


def xyz_to_csv(path):

    coords = np.zeros([0,3])

    with open(path,'r') as infile:        
        line = infile.readline()
        infile.readline()
        n_atoms = int(line)
        cnt = 0
        while(line):
            for i in range(n_atoms):
                coords = np.concatenate([coords,
                            np.array(infile.readline().split())[1:].astype(float).reshape(1,-1)])
            infile.readline()
            line = infile.readline()

    pd.DataFrame(coords).to_csv(path[:-3] + 'csv', index = None, header = None)
    
if __name__ == '__main__':
    xyz_to_csv('dimer.xyz')                
