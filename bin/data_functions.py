""" Specific functions for dataset treatment in pandas.

    These functions are written specifically to work in dataframes formatted for the flower plot
    project. Useful also as template to apply same functionality to different dataframes.
        * group_with_freq

    Authors: Juan Sebastian Diaz Boada
             juan.sebastian.diaz.boada@ki.se

    12/12/21
"""
import numpy as np
import pandas as pd
from collections import defaultdict
#---------------------------------------------------------------------------------------------------#
def group_with_freq(df,col,group_unique=False,new_name=None):
    """ Groups identical values and calculates their frequency, returning an updated dataframe.

        Calculates the frequency of each value in the column named 'col' from
        dataframe 'df', adding it into a column named 'freq_'+col. It also assigns a group number
        to each unique value in the column 'group_'+col. If the parameter 'group_unique'
        is True, it groups sequences that appear only once in one group labelled as -1. The suffix
        of the new column is by default the name of 'col', but can be changed adding a string as 
        'new_name' parameter.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with the column of values to group
        col : string
            Name of the column holding the values to analyze.
        group_unique : bool, optional
            Wether to group samples that only appear once in a group labelled '-1' or mantain each
            unique element as a separate group. Default is False.
        new_name : string, optional
            Suffix of the neame of the new column, instead of using the name of the old column.
            Default is None.

        Returns
        -------
        pd.DataFrame
            Dataframe with additional columns for group number and frequency.
    """
    DF = df.copy()
    freq_col_name = 'freq_' + col if new_name is None else 'freq_' + new_name
    group_col_name = 'group_'+col if new_name is None else 'group_' + new_name
    
    DF[freq_col_name] = DF[col].map(DF[col].value_counts()).astype(pd.Int64Dtype())
    DF.sort_values(by=[freq_col_name,col],ascending=False,inplace=True)
    if group_unique:
        # Default dict for cluster numbers. Return -1 if unseen instance
        seq2idx = defaultdict(lambda : -1)
        n_rep = len(DF.loc[DF[freq_col_name]!=1])
        seqs = DF.loc[:,col].iloc[:n_rep].unique()
    else:
        seq2idx = {}
        seqs = DF[col].unique()

    for n,g in enumerate(seqs):
        seq2idx[g]= n
    # Array with cluster numbers for each sequence
    group = np.array([seq2idx[i[col]] if not isinstance(i[col], float) \
                      else pd.NA for _,i in DF.iterrows()])
    DF[group_col_name]=group # Add cluster number to df
    return DF
#---------------------------------------------------------------------------------------------------#
