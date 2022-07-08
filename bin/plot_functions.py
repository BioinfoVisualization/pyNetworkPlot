""" Auxiliary functions in plot aesthetics.

    For organizing pie charts (matplotlib):
        * autopct_format
        
    For organizing subplots:
        * get_row_col
        * get_dimensions

    Authors: Juan Sebastian Diaz Boada - juan.sebastian.diaz.boada@ki.se
             SalvadorViramontes - https://stackoverflow.com/users/9873654/salvadorviramontes          
   19/01/22
"""

import numpy as np
from math import sqrt
#------------------------------------------------------------------------------------------#
def autopct_format(values):
    """ Function for showing counts rather than porcentages in pie charts.
        Taken from https://stackoverflow.com/a/53782750/8467949
        (Purely for visualization)
    """
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{v:d}'.format(v=val)
    return my_format
#------------------------------------------------------------------------------------------#
def get_row_col(i,n_cols):
    """ Returns the row and column of a matrix entry in plain form.
    
        Given a matrix entry as an integer i, returns the row and column assuming
        lexicographical ordering.
        
        Parameters
        ----------
        i : int
            Matrix index of the entry assuming the matrix as a flat array.
        n_cols : int
            Number of columns of the matrix.
            
        Returns
        -------
        tuple
            Tuple with two entries, the row index and the column index.
    """
    r = i//n_cols
    c = i-r*n_cols
    return r,c
#------------------------------------------------------------------------------------------#
def get_dimensions(i):
    """ Calculates the optimal dimensions of a rectangle grid with i entries.

        Parameters
        ----------
        i : int
            Number of entries of the rectangular grid.
            
        Returns
        -------
        np.array
            Array with two entries, the number of rows and the number of columns.
    
    """
    if i<=1: raise ValueError('The number of entries has to be greater than 1')
    s = int(np.ceil(sqrt(i)))
    S = np.array([(s,s),(s-1,s),(s-1,s+1)])
    S2 = np.array([S[i,0]*S[i,1] for i in range(len(S))])
    m = np.min(S2[S2-i>=0])
    return S[np.where(S2==m)[0][0]]