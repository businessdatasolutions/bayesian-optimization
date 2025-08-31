"""
Create static fallback charts for the interactive Bayesian Optimization demos
These will be used in the HTML slideshow export when widgets can't execute
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import plotly.io as pio

# Set plotly renderer for headless operation
try:
    pio.kaleido.scope.mathjax = None
except (AttributeError, TypeError):
    pass

def unknown_function(x):
    """The unknown function we're trying to optimize"""
    return -((x - 0.3) ** 2) + 0.5 * np.sin(15 * x) + 0.2

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """Expected Improvement acquisition function"""
    with np.errstate(divide='warn'):
        imp = mu - f_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def create_gp_plot(X_sample, y_sample, title_suffix=""):
    """Create a Gaussian Process visualization"""
    # Create GP model
    kernel = RBF(length_scale=0.2)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
    gp.fit(X_sample.reshape(-1, 1), y_sample)
    
    # Create prediction grid
    X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    y_plot = unknown_function(X_plot.ravel())
    mu, sigma = gp.predict(X_plot, return_std=True)
    
    # Create figure
    fig = go.Figure()
    
    # True function (hidden in real scenario)
    fig.add_trace(go.Scatter(
        x=X_plot.ravel(),
        y=y_plot,
        mode='lines',
        name='True Function (Hidden)',
        line=dict(color='red', dash='dash'),
        opacity=0.7
    ))
    
    # GP mean prediction
    fig.add_trace(go.Scatter(
        x=X_plot.ravel(),
        y=mu,
        mode='lines',
        name='GP Mean',
        line=dict(color='blue', width=2)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([X_plot.ravel(), X_plot.ravel()[::-1]]),
        y=np.concatenate([mu + 2*sigma, (mu - 2*sigma)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        showlegend=True
    ))
    
    # Observed points
    fig.add_trace(go.Scatter(
        x=X_sample,
        y=y_sample,
        mode='markers',
        name='Observed Points',
        marker=dict(color='black', size=8, symbol='circle')
    ))
    
    fig.update_layout(
        title=f'Gaussian Process Regression{title_suffix}',
        xaxis_title='Input (x)',
        yaxis_title='Output (y)',
        height=400,
        font=dict(size=12),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig

def create_bo_plot(X_sample, y_sample, title_suffix=""):
    """Create Bayesian Optimization visualization with acquisition function"""
    # Create GP model
    kernel = RBF(length_scale=0.2)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
    gp.fit(X_sample.reshape(-1, 1), y_sample)
    
    # Create prediction grid
    X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    y_plot = unknown_function(X_plot.ravel())
    mu, sigma = gp.predict(X_plot, return_std=True)
    
    # Calculate acquisition function
    f_best = np.max(y_sample)
    ei = expected_improvement(mu, sigma, f_best)
    
    # Create subplot with two y-axes
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Surrogate Model', 'Acquisition Function (Expected Improvement)'],
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Top plot: GP model
    fig.add_trace(go.Scatter(
        x=X_plot.ravel(),
        y=y_plot,
        mode='lines',
        name='True Function',
        line=dict(color='red', dash='dash'),
        opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=X_plot.ravel(),
        y=mu,
        mode='lines',
        name='GP Mean',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([X_plot.ravel(), X_plot.ravel()[::-1]]),
        y=np.concatenate([mu + 2*sigma, (mu - 2*sigma)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=X_sample,
        y=y_sample,
        mode='markers',
        name='Observed Points',
        marker=dict(color='black', size=8)
    ), row=1, col=1)
    
    # Bottom plot: Acquisition function
    fig.add_trace(go.Scatter(
        x=X_plot.ravel(),
        y=ei,
        mode='lines',
        name='Expected Improvement',
        line=dict(color='green', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,150,0,0.3)'
    ), row=2, col=1)
    
    # Find next point to sample
    next_x = X_plot[np.argmax(ei)][0]
    fig.add_trace(go.Scatter(
        x=[next_x],
        y=[np.max(ei)],
        mode='markers',
        name='Next Sample',
        marker=dict(color='red', size=10, symbol='star')
    ), row=2, col=1)
    
    fig.update_layout(
        title=f'Bayesian Optimization{title_suffix}',
        height=500,
        font=dict(size=11),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text='Input (x)', row=2, col=1)
    fig.update_yaxes(title_text='Output (y)', row=1, col=1)
    fig.update_yaxes(title_text='EI Value', row=2, col=1)
    
    return fig

def create_all_static_charts():
    """Create all static charts for different demo states"""
    
    # Initial state: 2 random points
    np.random.seed(42)
    X_initial = np.array([0.2, 0.8])
    y_initial = unknown_function(X_initial)
    
    # After 3 iterations
    X_3iter = np.array([0.2, 0.8, 0.35, 0.55, 0.15])
    y_3iter = unknown_function(X_3iter)
    
    # After 5 iterations (more exploration)
    X_5iter = np.array([0.2, 0.8, 0.35, 0.55, 0.15, 0.65, 0.25, 0.45])
    y_5iter = unknown_function(X_5iter)
    
    # Create and save charts
    charts = {
        'gp_initial': create_gp_plot(X_initial, y_initial, ' - Initial State'),
        'gp_3iter': create_gp_plot(X_3iter, y_3iter, ' - After 3 Iterations'),
        'gp_5iter': create_gp_plot(X_5iter, y_5iter, ' - After 5 Iterations'),
        'bo_initial': create_bo_plot(X_initial, y_initial, ' - Initial State'),
        'bo_3iter': create_bo_plot(X_3iter, y_3iter, ' - After 3 Iterations'),
        'bo_5iter': create_bo_plot(X_5iter, y_5iter, ' - After 5 Iterations')
    }
    
    # Save as HTML files
    for name, fig in charts.items():
        filename = f"static_{name}.html"
        fig.write_html(filename, include_plotlyjs='cdn', div_id=f"plotly_{name}")
        print(f"Created static chart: {filename}")
    
    # Also create a simple comparison chart
    create_comparison_chart()

def create_comparison_chart():
    """Create a comparison showing the optimization process"""
    x = np.linspace(0, 1, 200)
    y_true = unknown_function(x)
    
    # Show different sampling strategies
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Random Sampling', 'Grid Sampling', 'Bayesian Optimization'],
        horizontal_spacing=0.08
    )
    
    # Random sampling
    np.random.seed(42)
    x_random = np.random.uniform(0, 1, 5)
    y_random = unknown_function(x_random)
    
    fig.add_trace(go.Scatter(x=x, y=y_true, mode='lines', name='True Function', 
                            line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_random, y=y_random, mode='markers', name='Random Points',
                            marker=dict(color='blue', size=8)), row=1, col=1)
    
    # Grid sampling
    x_grid = np.linspace(0.1, 0.9, 5)
    y_grid = unknown_function(x_grid)
    
    fig.add_trace(go.Scatter(x=x, y=y_true, mode='lines', name='True Function', 
                            line=dict(color='red'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_grid, y=y_grid, mode='markers', name='Grid Points',
                            marker=dict(color='green', size=8)), row=1, col=2)
    
    # Bayesian optimization (focused around optimum)
    x_bo = np.array([0.2, 0.8, 0.35, 0.28, 0.32])
    y_bo = unknown_function(x_bo)
    
    fig.add_trace(go.Scatter(x=x, y=y_true, mode='lines', name='True Function', 
                            line=dict(color='red'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=x_bo, y=y_bo, mode='markers', name='BO Points',
                            marker=dict(color='orange', size=8)), row=1, col=3)
    
    fig.update_layout(
        title='Comparison of Sampling Strategies',
        height=300,
        font=dict(size=11),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(title_text='Input (x)')
    fig.update_yaxes(title_text='Output (y)')
    
    fig.write_html('static_comparison.html', include_plotlyjs='cdn', div_id="plotly_comparison")
    print("Created comparison chart: static_comparison.html")

if __name__ == "__main__":
    print("Creating static fallback charts...")
    create_all_static_charts()
    print("All static charts created successfully!")