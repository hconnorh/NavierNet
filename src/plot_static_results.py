#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import plotly.express as px


def plot_column(sim_name: str, col: str, case_name: int| None = None, 
                             step: int | None = None) -> None:
	"""
    Scatter plot x/y colored by the requested column at a given step.

	- column: name from the CSV header (e.g., "p", "u", "v", "vn").
	- sim_name: name of the simulation (used to locate the CSV file).
	- step: specific step to filter; if None, uses the last available step.
	- output_html: optional path to write HTML; if None, writes next to CSV.
	"""
	# Read the results CSV for the given simulation and case
	df = pd.read_csv(f"sims/{sim_name}/training_data/{case_name}-results.csv")

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
		color_continuous_scale="Viridis",
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
	fig.show()


if __name__ == "__main__":

    sim_name = "2d-cylinder-v1"

    case = "case0"
    step = 0
    col = "p"

    plot_column(sim_name, col='p', case_name=case, step=step)
