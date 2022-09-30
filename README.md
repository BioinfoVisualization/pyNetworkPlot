# pyNetworkPlot
Network plot for visualization of immune cells receptor clonality.

`pyNetworkPlot` is a small Python module build over [`igraph`]() and [`matplotlib`] that creates a Network plot from a sequence dataset. Ideal for clonality analysis of T cell/ B cell receptors (TCR/BRC), but can be extended to any analysis involving frequency of Strings.

## 1. Installation
`pyNetworkPlot` can be used in 3 different ways.
### 1.1 Installation of packages in [requirements.txt](requirements.txt)
`pyNetworkPlot` is tested in Python 3.9 and its dependencies can be pip-installed:
```bash
pip install -r requirements.txt
```
### 1.2 Setting a conda environment
This repository contains a conda definition file [`env/env.yml`](env/env.yml) that can create the conda environment:
```bash
conda env create -f env.yml
conda activate network
```
### 1.3 Singularity container
For Linux users, `pyNetworkPlot` can be run easily with a Singularity container. The container can be built with the definition file [`pyNetworkPlot.def`](env/pyNetwork.def) and run as follows:
```bash
sudo singularity build pyNetwork.sif containers/pyNetwork.def
./pyNetwork.sif <module_parameters>
```

## Usage
`pyNetworkPlot` can be used as a command-line tool running [`pyNetworkPlot.py`](src/pyNetworkPlot.py)
### Using the command line
By calling the script with the corresponding parameters in the correct environment, the script will save the plot in the location given by the user and in the given format.
```bash
./pyNetworkPlot.py in_file out_file --seq_col Sequence --color_col Color --shape_col Shape
               --size_col Shape --similarity 0 -custom_color /path/to/file --layout FR
               --use-legend
```
#### Parameters for command-line calling
+ `in_file` : string. Path to the DE dataset. The dataset must at least one column with the sequences. Additional columns for color, shape and size are optional.

|Sequence | Color | shape|
|------------ | ------------- | -------------|
|GATTACCA |#ff6db6 | 1|
|GATTACAA |#ff6db6 | 2|
... | ... | ...|


+ `out_path` : string. Path to the file where the plot is going to be saved.
+ `seq_col` : string (optional). Name of the column corresponding to the sequence in the dataset. Defaults to 'sequence'.
+ `color_col`  : string (optional). Name of the column corresponding to the color values in the dataset. Defaults to 'color'.
+ `shape_col` : string (optional). Name of the column corresponding to the shape values in the  dataset. Defaults to 'shape'.
+ `size_col` : string (optional). Name of the column corresponding to the size values in the dataset. Defaults to None.
+ `similarity` : int (optional). Maximum difference in amino acids between sequences to consider them similar. If non-zero, identical sequences will be plotted red and
similar sequences black. Defaults to zero (only plots edges between identical sequences).
+ `custom_color` : string. Path to a file mapping elements of color_col to a hex color code. This file has one line per unique value of column color_col. Each line
starts with the value, followed by a comma and the hex code for the color corresponding to that value. Defaults to None (use system default colors).
+ `layout` : string (optional). Keyword of the drawing algorithm to use. The options are 'FR' (Fruchterman-Reingold), 'DH' (Davidson-Harel), 'GO' (Graphopt),  DrL
(Dr Layout), LgL (Large Graph Layout) or  MDS (Multi-dimensional Scaling). Defaults to 'FR'.
+ `use-legend`  : flag. Include this flag to include a legend in the figure.
