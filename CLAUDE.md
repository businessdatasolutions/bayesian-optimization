# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains educational materials demonstrating Bayesian Optimization concepts for business managers in the logistics sector. The project consists of interactive Jupyter notebooks that explain optimization concepts through visual demonstrations and interactive widgets.

## Key Components

- **Interactive Notebooks**: Two main Jupyter notebooks (`test.ipynb` and `test2.ipynb`) containing Bayesian Optimization educational content
- **Presentation Materials**: HTML slide export (`test2.slides.html`) for presentation purposes
- **Virtual Environment**: Python environment managed in `.myenv/` directory with scientific computing dependencies

## Development Environment

### Python Setup
- **Python Version**: 3.12.2
- **Virtual Environment**: Located in `.myenv/` directory
- **Activation**: `source .myenv/bin/activate`

### Core Dependencies
The project uses scientific Python libraries for optimization and visualization:
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms (Gaussian Process Regressor)
- `scipy` - Statistical functions
- `plotly` - Interactive visualizations
- `ipywidgets` - Jupyter notebook widgets for interactivity

### Development Commands
```bash
# Activate virtual environment
source .myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter (if jupyter server components are installed)
jupyter notebook

# Work with notebooks interactively
# Open test.ipynb or test2.ipynb in Jupyter environment
```

## Notebook Architecture

The notebooks implement interactive demonstrations of:

1. **Gaussian Process Regression**: "Smart Predictor" that models unknown functions and quantifies uncertainty
2. **Acquisition Functions**: "Smart Decision Maker" that balances exploration vs exploitation
3. **Bayesian Optimization Loop**: Complete optimization cycle with interactive widgets

### Interactive Components
- Click-to-add data points on Gaussian Process visualizations
- Button-driven Bayesian Optimization iterations
- Real-time plot updates showing model predictions and uncertainties
- Dual-plot displays showing both surrogate model and acquisition function

## Educational Context

The notebooks are designed for business managers without mathematical backgrounds, focusing on:
- Logistics optimization scenarios (route planning, warehouse efficiency)
- Visual, intuitive explanations of complex optimization concepts
- Interactive demos to build understanding through hands-on experience
- Practical applications in supply chain and logistics operations