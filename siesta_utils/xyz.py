import numpy as np 
import pandas as pd


def xyz_to_csv(path, save_energies = False):

    coords = np.zeros([0,3])
    energies = []

    with open(path,'r') as infile:        
        line = infile.readline()
        


        n_atoms = int(line)
        line = infile.readline()
        cnt = 0
        while(line):
            print(line)
            energies.append(float(line))
            for i in range(n_atoms):
                coords = np.concatenate([coords,
                            np.array(infile.readline().split())[1:].astype(float).reshape(1,-1)])
            infile.readline()
            line = infile.readline()

    pd.DataFrame(coords).to_csv(path[:-3] + 'csv', index = None, header = None)
    if save_energies:
        pd.DataFrame(np.array(energies)).to_csv(path[:-3] + 'energies',
         index = None, header = None)

def csv_to_xyz_water(path, n_mol):
    
    coords = np.genfromtxt(path, delimiter = ',')
    n_entries = int(len(coords) / (n_mol*3))
    labels = ['O','H','H']*n_mol    
    cnt = 0

    with open(path[:-3]+'xyz','w') as outfile:
        
        for entry in range(n_entries):
            outfile.write(str(n_mol*3))
            outfile.write('\n \n')
            for l in labels:
                string = l + '\t'
                for i in range(3):
                    string += str(coords[cnt,i]) + '\t'
                outfile.write(string + '\n')
                cnt += 1 
if __name__ == '__main__':
    xyz_to_csv('dimer.xyz')                
