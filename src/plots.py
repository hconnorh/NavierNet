#!/usr/bin/env python3

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


"""
Module: plots.py
Package: NavierNet
Author: @hconnorh
Description: 

Utilities for plotting results from both CFD and surrogate simulation.

Key Features:
    - Loads case-specific extracted CSV files from a simulation folder
    - Visualises fields (pressure, velocity components, etc.) over nodes
"""


# Best with VS-CODE Monokai Pro theme
VS_PALLET = {
    'blue'   : '#4C55B7',
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

ELECTRIC_COLORSCALE = [
    [0.0, 'rgb(0, 0, 0)'],  # black
    [0.37695419864113106, 'rgb(2, 16, 83)'], 
    [0.559445018580004, 'rgb(9, 59, 157)'], 
    [0.7490160293761956, 'rgb(22, 115, 221)'], 
    [0.9068800985070032, 'rgb(73, 174, 243)'], 
    [1.0, 'rgb(255, 255, 255)']  # white
]

ELECTRIC_COLORSCALE_R = [[1.0 - p, c] for p, c in ELECTRIC_COLORSCALE[::-1]]

CMAP = {
	"electric": ELECTRIC_COLORSCALE,
	"electric_r": ELECTRIC_COLORSCALE_R,
	"turbo": "Turbo",
	"viridis": "Viridis",
}

def _ensure_pandas_sorted(df):
	"""Internal helper: accept Polars or Pandas and sort by epoch ascending."""
	if isinstance(df, pl.DataFrame):
		df = df.to_pandas()
	df = df.sort_values("epoch").reset_index(drop=True)
	return df


def plot_column(df: str, col: str, step: int | None = None, title: str = None,
			    theme: str ="turbo", plot_type="scatter") -> None:
	"""
	Scatter or contour plot of x/y colored by the requested column at a given step
	.
	Args:
		df (pandas DataFrame):	 pandas DataFrame with the data to plot.
		col (str): name from the CSV header (e.g., "p", "u", "v", "vn").
		step (int | None): specific step to filter; if None, uses the last available step.
		theme (str): key into CMAP (e.g., "turbo", "viridis", "electric", "electric_r").
		plot_type (str): "scatter" to show nodes, or "contour" to show filled contours.
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

	# Tight axis ranges to data bounds (handle degenerate ranges)
	xmin = float(np.min(x))
	xmax = float(np.max(x))
	ymin = float(np.min(y))
	ymax = float(np.max(y))
	if xmax == xmin:
		xmin -= 1.0
		xmax += 1.0
	if ymax == ymin:
		ymin -= 1.0
		ymax += 1.0

	if title is None:
		title = f"{col} over x/y" + (f" (step {step})" if step is not None else "")
	else: 
		title = f"{title} (step {step})"
		
	if plot_type == "scatter":
		fig = px.scatter(
			x=x,
			y=y,
			color=color_vals,
			color_continuous_scale=CMAP[theme],
			render_mode="webgl",
			labels={"color": col},
			title=title,
			range_x=[xmin, xmax],
			range_y=[ymin, ymax],
		)
		fig.update_traces(marker=dict(size=4))
		fig.update_layout(
			coloraxis_colorbar_title_text=col,
			width=900,
			# height=450,
			margin=dict(l=40, r=40, t=60, b=40)
		)
		fig.update_xaxes(scaleanchor="y", scaleratio=1)
		# Note: avoid equal-aspect constraint so axes match data min/max exactly
		return fig

	elif plot_type == "contour":
		# Create a regular grid over the domain and aggregate values into cells,
		# then fill sparse gaps by averaging neighboring cells for cleaner contours.
		xspan = float(xmax - xmin)
		yspan = float(ymax - ymin)
		Nx = 64
		Ny = max(64, int(round(Nx * (yspan / xspan)))) if xspan > 0 else Nx
		xedges = np.linspace(xmin, xmax, Nx + 1)
		yedges = np.linspace(ymin, ymax, Ny + 1)
		xcenters = 0.5 * (xedges[:-1] + xedges[1:])
		ycenters = 0.5 * (yedges[:-1] + yedges[1:])

		# Assign each point to a grid cell
		xv = x.to_numpy()
		yv = y.to_numpy()
		zv = vals.to_numpy().astype(float)
		ix = np.clip(np.searchsorted(xedges, xv, side='right') - 1, 0, Nx - 1)
		iy = np.clip(np.searchsorted(yedges, yv, side='right') - 1, 0, Ny - 1)

		# Accumulate per-cell sums and counts, then take mean
		sum_grid = np.zeros((Ny, Nx), dtype=float)
		cnt_grid = np.zeros((Ny, Nx), dtype=np.int64)
		np.add.at(sum_grid, (iy, ix), zv)
		np.add.at(cnt_grid, (iy, ix), 1)
		zgrid = np.divide(sum_grid, cnt_grid, out=np.full_like(sum_grid, np.nan, dtype=float), where=cnt_grid > 0)

		# Simple interpolation: iteratively fill NaNs using mean of 3x3 neighbors
		def _fill_nan_with_neighbors(z: np.ndarray, max_iter: int = 3) -> np.ndarray:
			filled = z.copy()
			for _ in range(max_iter):
				mask_nan = np.isnan(filled)
				if not np.any(mask_nan):
					break
				sums = np.zeros_like(filled, dtype=float)
				counts = np.zeros_like(filled, dtype=np.int64)
				for dy in (-1, 0, 1):
					for dx in (-1, 0, 1):
						if dy == 0 and dx == 0:
							continue
						y_src = slice(max(0, -dy), filled.shape[0] - max(0, dy))
						x_src = slice(max(0, -dx), filled.shape[1] - max(0, dx))
						y_dst = slice(max(0, dy), filled.shape[0] - max(0, -dy))
						x_dst = slice(max(0, dx), filled.shape[1] - max(0, -dx))
						nei = filled[y_src, x_src]
						m = ~np.isnan(nei)
						sums[y_dst, x_dst] += np.where(m, nei, 0.0)
						counts[y_dst, x_dst] += m.astype(np.int64)
				with np.errstate(invalid='ignore'):
					means = sums / counts
					fillable = mask_nan & (counts > 0)
					filled[fillable] = means[fillable]
			return filled

		zgrid_filled = _fill_nan_with_neighbors(zgrid, max_iter=3)

		trace = go.Contour(
			x=xcenters,
			y=ycenters,
			z=zgrid_filled,
			colorscale=CMAP[theme],
			zmin=float(vmin),
			zmax=float(vmax),
			ncontours=25,
			connectgaps=True,
			colorbar=dict(title=col),
		)
		fig = go.Figure(data=[trace])
		fig.update_layout(
			title=title,
			width=900,
			margin=dict(l=40, r=40, t=60, b=40)
		)
		fig.update_xaxes(range=[xmin, xmax])
		fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1)
		return fig

	else:
		raise ValueError("plot_type must be either 'scatter' or 'contour'")


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


def plot_compare(df_sim, df_ml, df_res, metric, step, cmap="turbo", plot_type="scatter"):
	"""
	Plot comparison of simulation, surrogate, and residual predictions for a
	single timestep in a 3-row layout with a shared Turbo colorscale.

	Args:
		df_sim: DataFrame (pandas or polars) of simulation truth rows
		df_ml: DataFrame (pandas or polars) of surrogate rows
		df_res: DataFrame (pandas or polars) of residual rows
		metric (str): column to color by (e.g., "p", "u", "v", "vn")
		step (int): timestep to visualize
		cmap (str): color map to use (e.g., "turbo", "viridis", "electric", "electric_r")
		plot_type (str): "scatter" to show nodes, or "contour" to show filled contours.
	Returns:
		plotly.graph_objects.Figure
	"""
	# Normalize input dataframes
	def _ensure_pandas(dfx):
		if isinstance(dfx, pl.DataFrame):
			return dfx.to_pandas()
		return dfx

	df_sim = _ensure_pandas(df_sim)
	df_ml = _ensure_pandas(df_ml)
	df_res = _ensure_pandas(df_res)

	# Filter by step and validate
	def _pick(df):
		out = df[df["step"] == step]
		if len(out) == 0:
			raise SystemExit(f"No rows for step={step}")
		return out

	ds = _pick(df_sim)
	dm = _pick(df_ml)
	dr = _pick(df_res)

	# Compute tight global axis bounds across all three panels
	xmin = float(min(ds["n_x"].min(), dm["n_x"].min(), dr["n_x"].min()))
	xmax = float(max(ds["n_x"].max(), dm["n_x"].max(), dr["n_x"].max()))
	ymin = float(min(ds["n_y"].min(), dm["n_y"].min(), dr["n_y"].min()))
	ymax = float(max(ds["n_y"].max(), dm["n_y"].max(), dr["n_y"].max()))
	if xmax == xmin:
		xmin -= 1.0; xmax += 1.0
	if ymax == ymin:
		ymin -= 1.0; ymax += 1.0

	# Color range shared across all traces (ensures consistent bar)
	def _range_for(arr: np.ndarray) -> tuple[float, float]:
		lo = float(np.min(arr)); hi = float(np.max(arr))
		if not np.isfinite(lo) or not np.isfinite(hi):
			raise SystemExit("Non-finite values encountered in metric column")
		if hi == lo:
			lo -= 1.0; hi += 1.0
		return lo, hi

	# Constrain color range 
	# NOTE: Currently disabled, can loose resolution across time.
	cmin_s, cmax_s = _range_for(ds[metric].to_numpy().astype(float))
	cmin_m, cmax_m = _range_for(dm[metric].to_numpy().astype(float))
	cmin_r, cmax_r = _range_for(dr[metric].to_numpy().astype(float))

	# Build subplot canvas
	fig = make_subplots(
		rows=3, cols=1, vertical_spacing=0.07,
		subplot_titles=["Truth (Sim)", "ML (Prediction)", "Residual (Abs Error)"]
	)

	def add_panel(dfp, rowi, axis_key: str):
		if plot_type == "scatter":
			fig.add_trace(
				go.Scattergl(
					x=dfp["n_x"], y=dfp["n_y"], mode="markers",
					marker=dict(
						size=4,
						color=dfp[metric].to_numpy(),
						coloraxis=axis_key,
					),
					showlegend=False,
				),
				row=rowi, col=1,
			)
		elif plot_type == "contour":
			# Contour: bin to a regular grid and lightly fill gaps for smoother fields
			def _grid_bin(df_local):
				xv = df_local["n_x"].to_numpy()
				yv = df_local["n_y"].to_numpy()
				zv = df_local[metric].to_numpy().astype(float)
				xspan = float(xmax - xmin)
				yspan = float(ymax - ymin)
				Nx = 64
				Ny = max(64, int(round(Nx * (yspan / xspan)))) if xspan > 0 else Nx
				xedges = np.linspace(xmin, xmax, Nx + 1)
				yedges = np.linspace(ymin, ymax, Ny + 1)
				xcenters = 0.5 * (xedges[:-1] + xedges[1:])
				ycenters = 0.5 * (yedges[:-1] + yedges[1:])
				ix = np.clip(np.searchsorted(xedges, xv, side='right') - 1, 0, Nx - 1)
				iy = np.clip(np.searchsorted(yedges, yv, side='right') - 1, 0, Ny - 1)
				sum_grid = np.zeros((Ny, Nx), dtype=float)
				cnt_grid = np.zeros((Ny, Nx), dtype=np.int64)
				np.add.at(sum_grid, (iy, ix), zv)
				np.add.at(cnt_grid, (iy, ix), 1)
				zgrid = np.divide(sum_grid, cnt_grid, out=np.full_like(sum_grid, np.nan, dtype=float), where=cnt_grid > 0)
				# Fill NaNs from neighbors (up to 3 iterations)
				filled = zgrid.copy()
				for _ in range(3):
					mask_nan = np.isnan(filled)
					if not np.any(mask_nan):
						break
					sums = np.zeros_like(filled, dtype=float)
					counts = np.zeros_like(filled, dtype=np.int64)
					for dy in (-1, 0, 1):
						for dx in (-1, 0, 1):
							if dy == 0 and dx == 0:
								continue
							y_src = slice(max(0, -dy), filled.shape[0] - max(0, dy))
							x_src = slice(max(0, -dx), filled.shape[1] - max(0, dx))
							y_dst = slice(max(0, dy), filled.shape[0] - max(0, -dy))
							x_dst = slice(max(0, dx), filled.shape[1] - max(0, -dx))
							nei = filled[y_src, x_src]
							m = ~np.isnan(nei)
							sums[y_dst, x_dst] += np.where(m, nei, 0.0)
							counts[y_dst, x_dst] += m.astype(np.int64)
					with np.errstate(invalid='ignore'):
						means = sums / counts
						fillable = mask_nan & (counts > 0)
						filled[fillable] = means[fillable]
				return xcenters, ycenters, filled

			xc, yc, zg = _grid_bin(dfp)
			fig.add_trace(
				go.Contour(
					x=xc,
					y=yc,
					z=zg,
					coloraxis=axis_key,
					ncontours=25,
					connectgaps=True,
					showscale=False,
				),
				row=rowi, col=1,
			)
		else:
			raise ValueError("plot_type must be either 'scatter' or 'contour'")
		# Tight and equal-aspect axes for this row
		fig.update_xaxes(range=[xmin, xmax], row=rowi, col=1)
		fig.update_yaxes(range=[ymin, ymax], scaleanchor=f"x{rowi}", scaleratio=1, row=rowi, col=1)

	# Add three panels
	add_panel(ds, 1, "coloraxis")
	add_panel(dm, 2, "coloraxis2")
	add_panel(dr, 3, "coloraxis3")

	# Position colorbars alongside each row domain
	# Retrieve y domains for rows 1..3
	ydom1 = fig.layout.yaxis.domain
	ydom2 = fig.layout.yaxis2.domain
	ydom3 = fig.layout.yaxis3.domain

	fig.update_layout(
		title=f"Nodal Residual Analysis ({metric}) | Step {step}",
		width=900,
		height=900*1.4,
		showlegend=False,
		margin=dict(l=40, r=40, t=80, b=40),
		
		# Individual coloraxes
		coloraxis=dict(
			colorscale=CMAP[cmap], showscale=True, #cmin=cmin_s, cmax=cmax_s,
			colorbar=dict(y=(ydom1[0]+ydom1[1])/2.0, yanchor="middle", len=ydom1[1]-ydom1[0])
		),
		coloraxis2=dict(
			colorscale=CMAP[cmap], showscale=True, #cmin=cmin_m, cmax=cmax_m,
			colorbar=dict(y=(ydom2[0]+ydom2[1])/2.0, yanchor="middle", len=ydom2[1]-ydom2[0])
		),
		coloraxis3=dict(
			colorscale=CMAP[cmap], showscale=True, #cmin=cmin_r, cmax=cmax_r,
			colorbar=dict(y=(ydom3[0]+ydom3[1])/2.0, yanchor="middle", len=ydom3[1]-ydom3[0])
		),
	)

	return fig