# Interstitial Plots Package

The Interstitial Plots package is a Python tool for visualizing and analyzing the proportion of dimension-related words present in actor texts. This package allows users to generate interactive plots using Plotly to represent the proportions of dimension words for different actors. It is particularly useful for understanding the distribution of specific dimension-related words in a corpus of text data.

## Installation

`pip install interstitial_plots_package`


## Usage
1. Using method in python code
```python
from src.pipeline import create_interstitial_plot

# Path to the JSON files
actor_dict_file = 'path/to/actor_dict.json'
dimension_dict_file = 'path/to/dimension_dict.json'

# Output folder to save the plot
output_folder = 'output_plots/'

# Create the interstitial plot
create_interstitial_plot(actor_dict_file, dimension_dict_file, output_folder)

```

2. Using CLI

```bash
$ ati actor_dict.json dimension_dict.json output_plots/
```