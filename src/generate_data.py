#!/usr/bin/env python3
# coding: utf-8
# =========================================================================== #
# generate_data.py                                                            #
# Author: Juan Sebastian Diaz Boada                                           #
# Creation Date: 25/10/22                                                     #
# =========================================================================== #
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('N', type=int, help="Number of samples.")
N = args.N
# Settings
nuc = ['A','C','G','T']
shapes = ['PB','BM','MUSL','LV']
colors = list(range(1,10))
sizes = [3,6,7,8,9,15]
# Creation of sequences
n = int(N/3)
lengths = np.random.randint(12,24,n)
sequences = ["".join(x for x in np.random.choice(nuc,lengths[i])) for i in range(n)]
samples = [np.random.choice(sequences) for i in range(N)]
# Dataframe
color_col = np.random.choice(colors,N)
shape_col = np.random.choice(shapes,N)
size_col = np.random.choice(sizes,N)
DF = pd.DataFrame({'sequence':samples,'color':color_col,'shape':shape_col,'size':size_col})
# Export
DF.to_csv('../data/test_data.csv',sep=',')
DF.to_csv('../data/test_data.tsv',sep='\t')
DF.to_excel('../data/test_data.xlsx')
