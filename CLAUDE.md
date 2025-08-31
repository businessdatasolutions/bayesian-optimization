# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains educational materials demonstrating Bayesian Optimization concepts for business managers in the logistics sector. The project uses Quarto to generate an interactive presentation that explains optimization concepts through visual demonstrations.

## Key Components

- **slides.qmd**: Main Quarto document containing the presentation source
- **_quarto.yml**: Quarto configuration for generating reveal.js presentations
- **requirements.txt**: Python dependencies for running the embedded code
- **Virtual Environment**: Python environment managed in `.myenv/` directory

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
- `plotly` - Interactive visualizations for charts

### Development Commands
```bash
# Activate virtual environment
source .myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate presentation
quarto render slides.qmd

# Preview during development
quarto preview slides.qmd
```

## Presentation Architecture

The Quarto presentation implements visual demonstrations of:

1. **Gaussian Process Regression**: "Smart Predictor" that models unknown functions and quantifies uncertainty
2. **Acquisition Functions**: "Smart Decision Maker" that balances exploration vs exploitation
3. **Bayesian Optimization Process**: Complete 4-stage learning progression visualization

### Visualization Components
- Static Plotly charts showing optimization progression
- Multi-panel figures demonstrating algorithm evolution
- Comparative charts showing different sampling strategies
- Business-focused explanations with logistics examples

## Educational Context

The presentation is designed for business managers without mathematical backgrounds, focusing on:
- Logistics optimization scenarios (route planning, warehouse efficiency)
- Visual, intuitive explanations of complex optimization concepts
- Progressive learning through 4-stage visualization
- Practical applications in supply chain and logistics operations