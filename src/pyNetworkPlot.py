#!/usr/bin/env python
# coding: utf-8
# =========================================================================== #
# pyNetworkPlot.py                                                            #
# Author: Juan Sebastian Diaz Boada                                           #
# Creation Date: 08/07/22                                                     #
# =========================================================================== #

""" Creates a network plot from a dataset and exports it.


    Parameters
    ----------
    in_path : string.
        Path to the sequence dataset.
    out_path : string.
        Path to the file where the plot is going to be saved.
    seq_col : string (optional).
        Name of the column corresponding to the sequencein the dataset.
        Defaults to 'sequence'.
    color_col : string (optional).
        Name of the column corresponding to the color values in the dataset.
        Defaults to 'color'.
    shape_col : string (optional).
        Name of the column corresponding to the shape values in the  dataset.
        Defaults to 'shape'.
    size_col : string (optional).
        Name of the column corresponding to the size values in the dataset.
        Defaults to None.
    layout : string (optional).
        Keyword of the drawing algorithm to use. The options are 'FR'
        (Fruchterman-Reingold), 'DH' (Davidson-Harel), 'GO' (Graphopt),  DrL
        (Dr Layout), LgL (Large Graph Layout) or  MDS (Multi-dimensional Scaling).
        Defaults to 'FR'.
    use-legend : flag.
        Include this flag to include a legend in the figure.

"""

import argparse
import warnings
import sys, os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from polyleven import levenshtein as poly_lev
#from collections import defaultdict
import igraph as ig
import random # layout seed
import cairo as cr
from igraph.drawing.text import TextDrawer
from math import pi # Legend circles
module_path = os.path.abspath(os.path.join('..', 'bin'))
if module_path not in sys.path:
    sys.path.append(module_path)
import spring_functions as kxs
from data_functions import group_with_freq

# Argparser definition
parser = argparse.ArgumentParser(description="Parameters of volcano plot.")
parser.add_argument('in_path', type=str, help="Path to the sequence dataset.")
parser.add_argument('out_path', type=str, help="Path to the file where the figure will be saved.")
parser.add_argument('--seq_col', type=str, default='sequence', help="Name of the column corresponding to the sequencein the dataset. Defaults to 'sequence'.")
parser.add_argument('--color_col', type=str, default=None, help="Name of the column corresponding to the color values in the dataset. Defaults to None.")
parser.add_argument('--shape_col', type=str, default=None, help="Name of the column corresponding to the shape values in the  dataset. Defaults to None.")
parser.add_argument('--size_col', type=str, default=None, help="Name of the column corresponding to the size values in the dataset. Defaults to None.")
parser.add_argument('--layout', type=str, default='FR', help="Keyword of the drawing algorithm to use. Defaults to 'FR'.")
parser.add_argument('--use-legend', dest='legend', action='store_true',help="Wether to include a legend in the figure.")
args = parser.parse_args()

# Data parameters
in_file = args.in_path
out_file = args.out_path
seq_col = args.seq_col
color_col = args.color_col
shape_col = args.shape_col
size_col = args.size_col
layout_name = args.layout
if color_col or shape_col or size_col:
    legend = args.legend
else:
    legend =False
    warnings.warn("Setting legend to False as all nodes are plotted equally.")

#min_seq2show = 0 # integer
group_unique = True # Boolean
similarity = 0 # non-negative integer
layout_name = 'FR' # Can be FR, DH, DrL, GO, LgL, MDS
unit=50
#edge_width = 1.5
max_node_size=50
min_node_size=5


# I. Data
file_type = in_file.split('.')[-1]
if file_type == 'tsv':
    DF = pd.read_csv(in_file,sep='\t',index_col=0).reset_index(drop=True)
elif file_type == 'xlsx':
    DF = pd.read_excel(in_file,index_col=0)
elif file_type == 'csv':
    DF = pd.read_csv(in_file,sep=',',index_col=0).reset_index(drop=True)
else:
    raise NameError("Invalid input format. Has to be either .tsv, .csv or .xlsx.")
DF = group_with_freq(DF,seq_col,group_unique).sort_values('freq_'+seq_col,ascending=False).reset_index(drop=True)
DF.loc[DF['group_'+seq_col]==-1,'group_'+seq_col]=DF['group_'+seq_col].max()+1

# II. Distance matrix calculation
seqs = DF[seq_col].values
L = len(seqs)
dist = np.zeros([L,L])
t = np.ceil(L/100)
for i in range(L):
    for j in range(L):
        dist[i,j]=poly_lev(seqs[i],seqs[j])
    if i%t==0:
        print("%.2f %% completed"%(i*100/L))
# Definite adjacency and weight matrices
eps = 0.1 # Distance delta
adj = dist.copy()
adj[adj<=similarity]=-1
adj[adj>similarity]=0
adj[adj==-1]=1
W = np.multiply(adj,dist+eps)
#plt.imshow(dist)
#plt.colorbar()

# III. Graph generation
# Create graph object
g = ig.Graph.Weighted_Adjacency(W,mode='undirected',attr='distance',loops=False)
# Node metadata
g.vs['cluster'] = DF.loc[:,'group_'+seq_col]
g.vs['freq'] = DF.loc[:,'freq_'+seq_col]
# Node color
if color_col == None:
    g.vs['color'] = 'red'
else:
    color_label = DF.loc[:,color_col].values
    _, idx = np.unique(color_label,return_index=True)
    labs = color_label[np.sort(idx)]
    n_labs = len(labs)
    pal = ig.drawing.colors.ClusterColoringPalette(n_labs)
    label2RGB = {l:pal.get_many(c)[0] for c,l in enumerate(labs)} # Numbering each label
    g.vs['color'] = [label2RGB[l] for l in color_label]
# Node shape
if shape_col == None:
    g.vs['shape'] = 'circle'
else:
    shapes = ['circle','rectangle','triangle-up','triangle-down','diamond']
    shape_labels = DF[shape_col].unique()
    n_shapes = len(shape_labels)
    if n_shapes > 5:
        raise ValueError('There can not be more than 5 shapes.')
    else:
        shapes = shapes[:n_shapes]
        shape_dic = {shape_labels[i]:shapes[i] for i in range(n_shapes)}
    g.vs['shape'] = DF.loc[:,shape_col].replace(shape_dic)
# Node size
if size_col==None:
    s = 20
else:
    s = DF[size_col].values
    s = (s-np.min(s))/(np.max(s)-np.min(s))*(max_node_size-min_node_size)+min_node_size
g.vs['size'] = s
# Graph layout
random.seed(42)
np.random.seed(42)
layout_seed = np.random.random([len(g.vs),2])
# Reingold-Fruchterman
if layout_name == 'FR':
    niter = 5000
    weights = kxs.prop_log_weights(g)
    g.es['weights'] = weights
    l = g.layout_fruchterman_reingold(weights=weights,
                                      seed=layout_seed,
                                      niter=niter)
# Davidson-Harel
elif layout_name == 'DH':
    maxiter = 80
    fineiter = 15
    cool_fact = 0.95
    weight_node_dist = 1000
    weight_border = 20000000
    weight_edge_lengths = 0.1
    weight_edge_crossings = 1000
    weight_node_edge_dist = 10000
    l = g.layout_davidson_harel(seed=layout_seed,
                                maxiter=maxiter,
                                fineiter=fineiter,
                                cool_fact=cool_fact,
                                weight_node_dist=weight_node_dist,
                                weight_border=weight_border,
                                weight_edge_lengths=weight_edge_lengths,
                                weight_edge_crossings=weight_edge_crossings,
                                 weight_node_edge_dist=weight_node_edge_dist)
# Graphopt
elif layout_name == 'GO':
    niter = 500
    node_charge = 0.03
    node_mass = 5
    spring_length = 5
    spring_constant = 0.5
    max_sa_movement = 12
    l = g.layout_graphopt(niter=niter, node_charge=node_charge,
                          node_mass=node_mass,
                          spring_length=spring_length,
                          spring_constant=spring_constant,
                          max_sa_movement=max_sa_movement,
                          seed=layout_seed)


# IV. Plot generation
if legend:
    label_h = 0.4*unit
    width,height = (24*unit,18*unit)

    # Construct the plot
    plot = ig.Plot(out_file, bbox=(width,height), background="white")
    plot.add(g, bbox=(1*unit, 1*unit, width-7*unit, height-1*unit),         vertex_size=g.vs['size'],layout=l)
    # Make the plot draw itself on the Cairo surface
    plot.redraw()
    # Grab the surface, construct a drawing context
    ctx = cr.Context(plot.surface)
    # Legend rectangle
    rect_height = label_h*len(label2RGB) + label_h
    rect_width = 3*unit # Change if the label is too long/short
    coord = [19*unit,9*unit-rect_height/2] # standing coordinates x,y

    ctx.rectangle(coord[0],coord[1], rect_width, rect_height)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(2)
    ctx.stroke()
    # Legend items
    coord[0]=coord[0]+label_h
    for l in label2RGB.keys():
        # Circle
        coord[1] = coord[1] + label_h
        ctx.move_to(coord[0],coord[1])
        ctx.arc(coord[0],coord[1], 0.1*unit, 0, 2*pi)
        ctx.close_path()
        ctx.set_source_rgb(label2RGB[l][0],label2RGB[l][1],label2RGB[l][2]) #R,G,B
        ctx.fill()
        # Text
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_font_size(0.3*unit)
        ctx.select_font_face("Arial",
                             cr.FONT_SLANT_NORMAL,
                             cr.FONT_WEIGHT_NORMAL)
        ctx.move_to(coord[0]+0.3*unit,coord[1]+0.1*unit)
        ctx.show_text(l)
    # Save the plot
    plot.save()
else:
    visual_style = {
        'bbox' : (0, 0, 600, 600),
        'layout' : l,
        "margin": 20,
        "autocurve" : False
        #'edge_width' : g.es['width'],
    }
    ig.plot(g,target=out_file,**visual_style)
