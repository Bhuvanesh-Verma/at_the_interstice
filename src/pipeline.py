import argparse
import json
import os
import re
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

from src.graph_plots import create_edge_trace, create_node_trace, create_network_graph


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text by words

    words = re.findall(r'\b\w+\b', text)
    return words

def preprocess_data(actor_dict=None, dimension_dict=None, actor_dict_file=None, dimension_dict_file=None):
    if actor_dict is None:
        with open(actor_dict_file) as f:
            actor_dict = json.load(f)

        with open(dimension_dict_file) as f:
            dimension_dict = json.load(f)

    preprocessed_actors = {}
    preprocessed_dimensions = {}
    shape_values = {}
    actor_rels = {}
    print("Preprocessing data ...")
    for actor, actor_data in tqdm(actor_dict.items()):
        preprocessed_actors[actor] = preprocess_text(actor_data['text'])
        shape_values[actor] = actor_data.get('shape', 'default_shape')
        actor_rels[actor] = actor_data.get('rel', [])

    for dim, words_list in dimension_dict.items():
        preprocessed_dimensions[dim] = [word.lower() for word in words_list]

    return preprocessed_actors, preprocessed_dimensions, shape_values, actor_rels

def count_word_occurrences(actor_texts, dimension_words):
    actor_word_counts = defaultdict(lambda: defaultdict(int))
    print('Comparing Text with dictionaries ...')
    for actor, words in tqdm(actor_texts.items()):
        for dim, dim_words in dimension_words.items():
            count = len(set(words).intersection(set(dim_words)))
            actor_word_counts[actor][dim] = count

    return actor_word_counts


def create_proportion_dataframe(actor_word_counts, dimension_words, shape_values, actor_rels):
    df_data = defaultdict(lambda: defaultdict())
    for actor, counts in actor_word_counts.items():
        for dim, count in counts.items():
            prop = count/len(dimension_words[dim])
            df_data[actor][dim] = prop
    actor_to_remove = []
    for actor, dim_data in df_data.items():
        total = sum(dim_data.values())
        if total == 0:
            actor_to_remove.append(actor)
            continue
        for dim, val in dim_data.items():
            df_data[actor][dim] = val/total
        df_data[actor]['shape'] = shape_values[actor]
        df_data[actor]['size'] = len(actor_rels[actor])
    for actor in actor_to_remove:
        del df_data[actor]
    print(f'Following actors have no match: {actor_to_remove}')
    df = pd.DataFrame(df_data).T
    return df
def get_all_shapes(size, type):
    all_shapes = []
    if type == '2d':
        raw_symbols = SymbolValidator().values
        if size < 54:
            for i in range(2, len(raw_symbols), 12):
                all_shapes.append(raw_symbols[i])
        else:
            for i in range(0, len(raw_symbols), 3):
                all_shapes.append(raw_symbols[i + 2])
    else:
        all_shapes = ['circle', 'x', 'diamond','square', 'circle-open', 'cross', 'diamond-open', 'square-open']

    return all_shapes

def prepare_plot_params(dataframe, actor_rels, shape_values, dimensions, type='2d'):
    actor_sizes = [len(actor_rels[actor]) + 10 for actor in dataframe.index]
    shapes = get_all_shapes(size=len(list(set(shape_values.values()))), type=type)
    shape_map = {val: shapes[i] for i, val in enumerate(list(set(shape_values.values())))}
    actor_shapes = [shape_map[shape_values[actor]] for actor in dataframe.index]

    # Calculate custom colors based on proportion values for each actor
    custom_colors = []
    for actor in dataframe.index:
        code = np.array([255, 255, 255])
        proportions = dataframe.loc[actor, dimensions].values
        custom_colors.append(f"rgb{tuple(np.array(code * np.array(proportions), dtype=int))}")

    return actor_sizes, actor_shapes, custom_colors, shape_map

def create_multi_dimensional_plot(dataframe, dimensions, shape_values, actor_rels):
    if len(dimensions) == 3:
        actor_sizes, actor_shapes, custom_colors, shape_map = prepare_plot_params(dataframe, actor_rels, shape_values,
                                                                                  dimensions, type='2d')

        fig = px.scatter_ternary(
            dataframe,
            a=dimensions[0],
            b=dimensions[1],
            c=dimensions[2],
            hover_name=dataframe.index,
            size=actor_sizes,  # Set the marker size based on 'rel' list length

        )
        fig.update_traces(
            marker=dict(
                symbol=actor_shapes,
                color=custom_colors,

            ),
            selector=dict(mode="markers"),
            legendgroup='actors',  # Group the actor markers in the legend
            name='',
        )

        # Add a custom legend trace for the shape mapping
        for shape_value, shape_name in shape_map.items():

            fig.add_trace(go.Scatterternary(
                a=[None],
                b=[None],
                c=[None],
                mode='markers',
                marker=dict(
                    symbol=shape_name,
                    #color='rgba(0,0,0,0)',  # Transparent color to hide the legend points
                ),
                name=shape_value,
                legendgroup='shapes',  # Group the shape legend items
                showlegend=True,
            ))

        # Update the layout of the figure to include the custom legend
        fig.update_layout(
            ternary=dict(sum=100, aaxis=dict(title=dimensions[0], ticksuffix='%'),
                         baxis=dict(title=dimensions[1], ticksuffix='%'),
                         caxis=dict(title=dimensions[2], ticksuffix='%')),
            showlegend=True,  # Enable the legend
            legend=dict(
                traceorder='grouped',  # To order legend items based on their group
                itemsizing='constant',  # To keep legend item sizes constant
            )
        )
    else:
        raise NotImplementedError("Dimensions other than 3 are not supported yet.")


    return fig


def create_3d_scatter_plot(dataframe, x_col, y_col, z_col):
    """
    Create a 3D scatter plot using Plotly.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The column name for x-coordinates.
        y_col (str): The column name for y-coordinates.
        z_col (str): The column name for z-coordinates.
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=dataframe[x_col],
        y=dataframe[y_col],
        z=dataframe[z_col],
        mode='markers',
        text=dataframe.index,  # Set the DataFrame index as the hover text
        hoverinfo='text',  # Show only the text defined above on hover
        marker=dict(
            size=10,
            colorscale='Viridis',  # Choose a colorscale (you can change this to any other available colorscale)
            opacity=0.8
        )
    )])

    # Set axis labels
    fig.update_layout(scene=dict(
        xaxis_title=x_col,
        yaxis_title=y_col,
        zaxis_title=z_col
    ))

    """# Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the plot as an interactive HTML file in the output folder
    output_file = os.path.join(output_folder, "3d_interstitial_plot.html")
    fig.write_html(output_file)

    print(f"3D Interstitial plot saved to {output_file}")"""

def create_manual_3d(df, dimensions, shape_values, actor_rels):
    actor_sizes, actor_shapes, custom_colors, shape_map = prepare_plot_params(df, actor_rels, shape_values,
                                                                              dimensions, type='3d')
    G =nx.Graph()
    hover_text = []

    for actor in df.index:
        G.add_node(actor, pos=df.loc[actor, dimensions].values)

        hover_text.append(f'{actor}')
    for actor, rels in actor_rels.items():
        for rel in rels:
            if rel in df.index and actor in df.index:
                G.add_edge(actor, rel)


    title = f'<br>3D Interstitial Plot'
    traces = create_edge_trace(G, dim=3)
    traces.append(create_node_trace(G, hover_text, actor_sizes, custom_colors,actor_shapes, dim=3))
    fig = create_network_graph(traces, title,shape_map,subjects=dimensions, dim=3)

    return fig


def create_interstitial_plot(plot_type,data=None, dim_dict=None,  actor_dict_file=None, dimension_dict_file=None,
                             output_dir=None, save=False):
    preprocessed_actors, preprocessed_dimensions, shape_values, actor_rels = preprocess_data(
        data, dim_dict, actor_dict_file, dimension_dict_file
    )
    actor_word_counts = count_word_occurrences(preprocessed_actors, preprocessed_dimensions)
    proportion_dataframe = create_proportion_dataframe(actor_word_counts, preprocessed_dimensions, shape_values, actor_rels)
    dimensions = list(preprocessed_dimensions.keys())
    if plot_type == '2d':
        # Create the 2D multi-dimensional plot
        fig = create_multi_dimensional_plot(proportion_dataframe, dimensions, shape_values, actor_rels)
        name = "2d_interstitial_plot.html"
    elif plot_type == '3d' and len(dimensions) == 3:
        # Create the 3D scatter plot using the first 3 columns of the DataFrame
        #create_3d_scatter_plot(proportion_dataframe, output_dir, dimensions[0], dimensions[1], dimensions[2])
        fig = create_manual_3d(proportion_dataframe, dimensions, shape_values, actor_rels)
        name = "3d_interstitial_plot.html"
    else:
        raise ValueError("Invalid plot type or dimensions for 3D plot.")

    if save:
        # Create the output folder if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        # Save the plot as an image file in the output folder
        output_file = os.path.join(output_dir, name)
        fig.write_html(output_file)
        print(f"Plot saved to {output_file}")

    return fig

def create_from_files(plot_type, actor_dict_file, dimension_dict_file, output_dir=None, save=False):
    create_interstitial_plot(plot_type, actor_dict_file, dimension_dict_file, output_dir, save)

def create_from_data(data, dimension_dict, plot_type,output_dir=None, save=False):
    create_interstitial_plot(data, dimension_dict, plot_type, output_dir, save)

def main():
    parser = argparse.ArgumentParser(description='Create a interstitial plot from input dictionaries.')
    parser.add_argument('-actor_dict_file', type=str, help='Path to the JSON file containing the actor dictionary.')
    parser.add_argument('-dimension_dict_file', type=str,
                        help='Path to the JSON file containing the dimension dictionary.')
    parser.add_argument('-output_folder', type=str, help='Folder to save the generated plots.')
    parser.add_argument('-save', type=bool, default=False, help='Folder to save the generated plots.')
    parser.add_argument('-plot_type', type=str, choices=['2d', '3d'], default='2d',
                        help='Type of plot: "2d" for 2D multi-dimensional plot, "3d" for 3D scatter plot.')
    args = parser.parse_args()

    create_from_files(args.actor_dict_file, args.dimension_dict_file, args.plot_type, args.output_folder, args.save)

if __name__ == '__main__':
    main()