#!/usr/bin/env python3
import os
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

"""
Module: plot_static_results.py
Package: NS2D-Surrogate
Author: @hconnorh
Description:

This script provides utility plotting functionality for static visualisation of
simulated nodal fields (e.g., pressure, velocity) from PyFR-generated training 
data CSVs.

It is used to generate static x/y scatter plots colorized by nodal property, 
for a specified timestep, case, and simulation. Intended for analysis and 
verification of simulation results.

Key Features:
    - Loads case-specific extracted CSV files from a simulation folder
    - Visualises fields (pressure, velocity components, etc.) over the mesh
    - Presents plots using Plotly web/webgl for large datasets

Date: 09-May-2023
Modified: 17-Sep-2025
"""

# Best with VS-CODE Monokai Pro theme
VS_PALLET = {
    'blue'   :  '#4C55B7',
    'red'    : '#AD4332',
    'green'  : '#079671',
    'purple' : '#7E4DB7',
    'orange' : '#B97848',
    'yellow' : '#F2D658',
    'pink'   : 'HSL(300,90,70)',
    'lblack' : '#161A1D',
    'black'  : '#0E1012',
    'grey'   : '#273240',
    'white'  : '#D7DBDF',
    'vs-grey-dark': 'HSL(315,6,13)',
    'vs-grey': 'RGBA(0,0,0,0)',
    'vs-grey-light': 'HSL(300,2,34)',
}


def _ensure_pandas_sorted(df):
	"""Internal helper: accept Polars or Pandas and sort by epoch ascending."""
	if isinstance(df, pl.DataFrame):
		df = df.to_pandas()
	df = df.sort_values("epoch").reset_index(drop=True)
	return df


def plot_column(df: str, col: str, case_name: int| None = None, 
                step: int | None = None, theme: str ="Turbo") -> None:
	"""
    Scatter plot x/y colored by the requested column at a given step.

	- column: name from the CSV header (e.g., "p", "u", "v", "vn").
	- sim_name: name of the simulation (used to locate the CSV file).
	- step: specific step to filter; if None, uses the last available step.
	- output_html: optional path to write HTML; if None, writes next to CSV.
	"""

	# Check if the requested step exists; if step is None, pick max step
	if step is None:
		step = df['step'].max()
	df = df[df['step'] == step]

	if len(df) == 0:
		raise SystemExit(f"No rows for step={step}")

	x = df["n_x"]
	y = df["n_y"]
	vals = df[col]

	# If values are constant, avoid invalid color scaling
	vmin = float(np.min(vals))
	vmax = float(np.max(vals))
	color_vals = vals if vmax > vmin else np.zeros_like(vals, dtype=float)
	title = f"{col} over x/y" + (f" (step {step})" if step is not None else "")
	fig = px.scatter(
		x=x,
		y=y,
		color=color_vals,
		color_continuous_scale=theme,
		render_mode="webgl",
		labels={"color": col},
		title=title
	)
	fig.update_traces(marker=dict(size=4))
	fig.update_layout(
		coloraxis_colorbar_title_text=col,
		width=900,
		height=500,
		margin=dict(l=40, r=40, t=60, b=40)
	)
	fig.update_yaxes(scaleanchor="x", scaleratio=1)

	return fig


def base_layout(title, xlabel, ylabel):
	fig = go.Figure()

	fig.update_layout(
		title=title,
		xaxis_title=xlabel,
		yaxis_title=ylabel,
		paper_bgcolor=VS_PALLET['vs-grey'],
		plot_bgcolor=VS_PALLET['vs-grey'],  
		font=dict(color=VS_PALLET['white']),
		xaxis=dict(
			gridcolor=VS_PALLET['vs-grey-light'], 
			gridwidth=1,
			showline=True,         
			linewidth=2,           
			linecolor=VS_PALLET['vs-grey-light'],     
			mirror=True,          
			zeroline=True,          
			zerolinewidth=1,        
			zerolinecolor=VS_PALLET['vs-grey-light'] 
		),
		yaxis=dict(
			gridcolor=VS_PALLET['vs-grey-light'], 
			gridwidth=1, 
			showline=True,        
			linewidth=2,          
			linecolor=VS_PALLET['vs-grey-light'],
			mirror=True,          
			zeroline=True,       
			zerolinewidth=1,     
			zerolinecolor=VS_PALLET['vs-grey-light'] 
		),
		legend=dict(
			orientation='h',       
			x=0,                  
			y=-0.15,                   
			yanchor='top', 
			bgcolor='RGBA(0,0,0,0)'
		),
		margin=dict(
			l=20,
			r=20,
			t=40,
			b=20
		)
	)
	return fig

def plot_loss(df):
	"""
	Plot train and validation loss over epochs.

	This function visualises the progression of training and validation loss 
	over epochs. It highlights the epoch with minimum validation loss. Useful 
	for examining model training history and selecting the best checkpoint.
	"""
	df = _ensure_pandas_sorted(df)

	# Set base layout
	fig = base_layout(title='Training Loss', xlabel='epochs', ylabel='loss [-]')

	# Add traces
	c_keys = list(VS_PALLET.keys())
	epochs = df["epoch"].to_numpy()
	for i, c in enumerate(['train_loss', 'val_loss']):
		data = df[c].to_numpy()
		fig.add_trace(
			go.Scatter(
			x=epochs, y=data, mode="lines+markers", name=c, 
			line=dict(color=VS_PALLET[c_keys[i]])
			)
		)

	# Add best epoch
	val_loss = df["val_loss"].to_numpy()
	best_idx = int(np.argmin(val_loss))
	best_epoch = int(epochs[best_idx])
	best_val_loss = float(val_loss[best_idx])
	fig.add_vline(x=best_epoch, line_width=1, 
				  line_dash="dot", line_color=VS_PALLET['pink'])
	fig.add_trace(
		go.Scatter(
			x=[best_epoch], y=[best_val_loss], mode="markers",
			name="best_epoch (val_loss)", marker=dict(color=VS_PALLET['pink'], 
			size=10, symbol="star")
		)
	)
	
	return fig


def plot_rmse(df):
	"""
	Plot validation RMSEs for du, dv, dp over training epochs.

	This function visualises the root mean squared errors (RMSE) for the 
	predicted changes in u-velocity (du), v-velocity (dv), and pressure (dp)
	on the validation set across training epochs. Each metric is plotted as a 
	separate line. The epoch with the minimum validation loss is highlighted 
	with a vertical dashed line.
	"""
	df = _ensure_pandas_sorted(df)

	# Set base layout
	fig = base_layout(
		title=f'RMSE (Pressure and Velocity components)',
		xlabel='epochs',
		ylabel='rmse [-]'
	)

	# Add traces
	c_keys = list(VS_PALLET.keys())
	epochs = df["epoch"].to_numpy()
	for i, c in enumerate(['val_rmse_du', 'val_rmse_dv', 'val_rmse_dp']):
		data = df[c].to_numpy()
		fig.add_trace(
			go.Scatter(
			x=epochs, y=data, mode="lines+markers", name=c, 
			line=dict(color=VS_PALLET[c_keys[i]])
			)
		)

	# Add best epoch
	best_idx = int(np.argmin(df["val_loss"].to_numpy()))
	best_epoch = int(epochs[best_idx])
	fig.add_vline(
		x=best_epoch, line_width=1, name="best_epoch (val_loss)",
		line_dash="dot", line_color=VS_PALLET['pink']
	)

	return fig


def plot_penalty(df):
    """
    Plot validation divergence and boundary penalties over epochs.

    This function visualises the evolution of divergence penalty (`val_div`) 
	and boundary penalty (`val_bnd`) on the validation set as training 
	progresses. These two metrics are used to assess the model's ability to 
	enforce incompressibility and boundary conditions respectively. As training 
	progresses, the model should be able to reduce these penalties thus 
	enforcing physical constraints within the training data.

    Both metrics are plotted as separate lines over the training epochs, 
	allowing you to assess the model's ability to enforce incompressibility 
	and boundary conditions. The epoch corresponding to the minimum validation 
	loss is highlighted with a vertical dashed line for reference.
    """
    df = _ensure_pandas_sorted(df)

    # Set base layout
    fig = base_layout(
        title='Incompressability Penalty',
        xlabel='epochs',
        ylabel='penalty [-]'
    )

    # Add traces
    c_keys = list(VS_PALLET.keys())
    epochs = df["epoch"].to_numpy()
    for i, c in enumerate(['val_div', 'val_bnd']):
        data = df[c].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=epochs, y=data, mode="lines+markers", name=c,
                line=dict(color=VS_PALLET[c_keys[i]])
            )
        )

    # Add best epoch
    best_idx = int(np.argmin(df["val_loss"].to_numpy()))
    best_epoch = int(epochs[best_idx])
    fig.add_vline(
        x=best_epoch, line_width=1, name="best_epoch (val_loss)",
        line_dash="dot", line_color=VS_PALLET['pink']
    )

    return fig


def plot_training_gap(df):
	"""
	Plot generalisation gap (train - val) and mean validation RMSE over epochs.

	This function visualises the generalisation gap (train - val) and mean 
	validation RMSE over training epochs. The generalisation gap is the 
	difference between the training loss and the validation loss. The mean 
	validation RMSE is the average of the RMSE for the predicted changes in 
	u-velocity (du), v-velocity (dv), and pressure (dp) across all nodes in the
	simulation domain.

	The generalisation gap is used to assess the model's ability to generalise 
	to unseen data or in other words, how "overfit" the model is.
	"""
	df = _ensure_pandas_sorted(df)

	# Set base layout
	fig = base_layout(
		title=f'Generalisation vs Mean RMSE',
		xlabel='epochs', 
		ylabel='rmse [-]'
	)

	# Calculate mean rmse and gap 
	df['val_rmse_mean'] = (df["val_rmse_du"] + df["val_rmse_dv"] + df["val_rmse_dp"]) / 3.0
	df['gen_gap'] = df["train_loss"] - df["val_loss"]

	# Add traces
	c_keys = list(VS_PALLET.keys())
	epochs = df["epoch"].to_numpy()
	for i, c in enumerate(['val_rmse_mean', 'gen_gap']):
		data = df[c].to_numpy()
		fig.add_trace(
			go.Scatter(
			x=epochs, y=data, mode="lines+markers", name=c, 
			line=dict(color=VS_PALLET[c_keys[i]])
			)
		)

	# Add best epoch
	best_idx = int(np.argmin(df["val_loss"].to_numpy()))
	best_epoch = int(epochs[best_idx])
	fig.add_vline(
		x=best_epoch, line_width=1, name="best_epoch (val_loss)",
		line_dash="dot", line_color=VS_PALLET['pink']
	)

	return fig
