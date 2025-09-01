title: "🎯 Slimmer Beslissen in Logistiek" subtitle: "Een Introductie tot Bayesian Optimization" author: "Witek ten Hove" format: revealjs: theme: white slide-number: true transition: slide background-transition: fade incremental: false width: 1600 height: 900 margin: 0.1 min-scale: 0.2 max-scale: 1.5 center: true html-math-method: katex embed-resources: true jupyter: python3 execute: freeze: auto cache: false
Welkom {.center}
Voor wie: De logistieke professional die vooruit wil.

Investering: 30 minuten van uw tijd.

Resultaat: Een interactieve ervaring die uw kijk op optimalisatie verandert.

<div style="text-align: center; font-size: 1.2em; color: #2E86AB; margin-top: 50px;">
<strong>Welkom bij de volgende stap in optimalisatie.</strong>
</div>

🎯 Wat Neemt u Mee?
Na dit halfuur bent u in staat om:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 30px;">

<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
<h3>🎯 Kansen Identificeren</h3>
<p>De kostbare 'trial-and-error'-processen in uw organisatie te herkennen en de kansen voor een slimmere aanpak te zien.</p>
</div>

<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #17a2b8;">
<h3>🚀 Sneller Resultaat</h3>
<p>De kracht te doorgronden van een methode die sneller tot betere resultaten leidt, met significant minder experimenten.</p>
</div>

<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;">
<h3>🗣️ De Juiste Vragen Stellen</h3>
<p>Een helder, strategisch gesprek te voeren met uw data scientists over de kernprincipes van Bayesian Optimization.</p>
</div>

<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;">
<h3>⚖️ Strategisch Afwegen</h3>
<p>De cruciale balans te begrijpen tussen het verkennen van nieuwe mogelijkheden en het benutten van bewezen successen.</p>
</div>

</div>

::: {.center}
Straks duiken we in de praktijk met interactieve demo's. Maak u klaar om zelf te ontdekken.
:::

📋 De Route van Vandaag
<div style="display: flex; align-items: center; justify-content: space-between; margin: 40px 0;">

<div style="flex: 1; text-align: center;">
<div style="background: #e3f2fd; border-radius: 50%; width: 80px; height: 80px; margin: 0 auto 15px; display: flex; align-items: center; justify-content: center; font-size: 2em;">🎯</div>
<strong>De Kern van de Uitdaging</strong><br/>
<small>2 minuten</small>
</div>

<div style="flex: 0 0 50px; text-align: center; font-size: 2em; color: #ccc;">→</div>

<div style="flex: 1; text-align: center;">
<div style="background: #f3e5f5; border-radius: 50%; width: 80px; height: 80px; margin: 0 auto 15px; display: flex; align-items: center; justify-content: center; font-size: 2em;">🧠</div>
<strong>Een Strategisch Antwoord</strong><br/>
<small>7 minuten</small>
</div>

<div style="flex: 0 0 50px; text-align: center; font-size: 2em; color: #ccc;">→</div>

<div style="flex: 1; text-align: center;">
<div style="background: #e8f5e8; border-radius: 50%; width: 80px; height: 80px; margin: 0 auto 15px; display: flex; align-items: center; justify-content: center; font-size: 2em;">🔄</div>
<strong>Het Leerproces in de Praktijk</strong><br/>
<small>4 minuten</small>
</div>

<div style="flex: 0 0 50px; text-align: center; font-size: 2em; color: #ccc;">→</div>

<div style="flex: 1; text-align: center;">
<div style="background: #fff3e0; border-radius: 50%; width: 80px; height: 80px; margin: 0 auto 15px; display: flex; align-items: center; justify-content: center; font-size: 2em;">🚛</div>
<strong>De Impact op uw Resultaat</strong><br/>
<small>2 minuten</small>
</div>

</div>

::: {.center style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-top: 30px;"}

🤖 De Hamvraag in de Logistiek
Hoe vinden we de optimale oplossing,

zonder een fortuin uit te geven aan experimenten?

:::

🎯 De Uitdaging: Kostbare 'Black Box'-Problemen
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 30px;">

<div>
<h2 style="color: #dc3545;">🔲 Wat is een 'Black Box'?</h2>
<div style="background: #f8d7da; padding: 20px; border-radius: 10px; border: 1px solid #f5c6cb;">
<p><strong>Stelt u zich voor:</strong> nieuwe software voor routeplanning.</p>
<ul>
<li>U voert parameters in (capaciteit, levertijden)</li>
<li>De software presenteert een efficiëntiescore</li>
<li>De complexe logica áchter die score blijft verborgen</li>
<li><strong>Het enige wat telt: INPUT → OUTPUT</strong></li>
</ul>
</div>
</div>

<div>
<h2 style="color: #fd7e14;">💰 Waarom is Experimenteren zo Kostbaar?</h2>
<div style="background: #ffeaa7; padding: 20px; border-radius: 10px; border: 1px solid #ffd93d;">
<p>Iedere test vraagt een investering in:</p>
<ul>
<li><strong>Tijd:</strong> Weken voor een pilot, maanden voor een project</li>
<li><strong>Geld:</strong> Brandstof, manuren, operationele verstoring</li>
<li><strong>Middelen:</strong> Voertuigen, personeel, rekenkracht</li>
</ul>
<p style="margin-top: 15px; font-weight: bold; color: #d63031;">❌ Duizenden scenario's testen is simpelweg onmogelijk.</p>
</div>
</div>

</div>

⚡ Het Dilemma
<div style="background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; margin-top: 40px;">
<h2>⚡ Het Dilemma</h2>
<p style="font-size: 1.3em; margin: 0;">
U streeft naar het ultieme resultaat, maar weet dat u nooit alle opties kunt testen.<br/>
Puur gokwerk is inefficiënt. <strong>Het is tijd voor een doorbraak: een slimmere strategie.</strong>
</p>
</div>

<div style="text-align: center; margin-top: 30px; font-size: 1.1em; color: #666;">
<strong>En die strategie? Die heet Bayesian Optimization. Laten we ontdekken hoe het werkt. →</strong>
</div>

🧠 Een Strategisch Antwoord: Bayesian Optimization
<div style="text-align: center; margin: 40px 0;">
<div style="background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); color: white; padding: 30px; border-radius: 15px;">
<h2 style="margin-top: 0;">🎭 De Twee Pilaren van een Zelflerend Systeem</h2>
<p style="font-size: 1.2em;">Vergelijk het met een expert die met elk experiment scherpere inzichten krijgt.</p>
</div>
</div>

<div style="display: grid; grid-template-columns: 1fr 50px 1fr; gap: 20px; align-items: center; margin-top: 40px;">

<div style="background: #dff0d8; padding: 25px; border-radius: 15px; border: 2px solid #d6e9c6; text-align: center;">
<h3 style="color: #3c763d; margin-top: 0;">🔮 De Intelligente Voorspeller</h3>
<p><strong>(Surrogaatmodel)</strong></p>
<p>Leert van eerdere resultaten en bouwt een aanname op over het totale speelveld.</p>
<p style="font-style: italic; color: #5a6c5d;">Net als een ervaren manager die feilloos aanvoelt welke routes potentie hebben.</p>
</div>

<div style="text-align: center; font-size: 3em; color: #a29bfe;">
⟷
</div>

<div style="background: #d9edf7; padding: 25px; border-radius: 15px; border: 2px solid #bce8f1; text-align: center;">
<h3 style="color: #31708f; margin-top: 0;">🎯 De Strategische Beslisser</h3>
<p><strong>(Acquisitiefunctie)</strong></p>
<p>Bepaalt welke volgende test de meest waardevolle informatie oplevert.</p>
<p style="font-style: italic; color: #5a6c7d;">Vindt de balans tussen het verkennen van onbekend terrein en het perfectioneren van wat al goed presteert.</p>
</div>

</div>

🎯 Van Onzekerheid naar Inzicht: Het Model in Actie
<div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 30px;">
<h2 style="margin: 0; font-size: 1.5em;">De Leerreis van Onzekerheid naar Zekerheid</h2>
<p style="font-size: 1.1em; margin: 10px 0 0 0;">Observeer hoe de 'Intelligente Voorspeller' met elk nieuw datapunt aan zekerheid wint.</p>
</div>

#| echo: false
#| output: true
#| label: fig-gp-progression
#| fig-width: 11
#| fig-height: 8
#| fig-align: center
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

def black_box_function(x):
    """The true function we're trying to optimize"""
    return np.sin(0.9 * x) * np.sinc(x * 0.2) * 50 + 25

# Setup
x_range = np.linspace(0, 20, 200).reshape(-1, 1)
y_true = black_box_function(x_range)

# Define sampling points for each iteration
initial_points = np.array([[2.0], [18.0]])
additional_points = [
    np.array([[10.0]]),
    np.array([[5.0]]),
    np.array([[14.0]]),
    np.array([[7.5], [12.0]]),
]

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '<b>Stap 1: Eerste Verkenning (2 observaties)</b>',
        '<b>Stap 2: Vroeg Leerproces (3 observaties)</b>',
        '<b>Stap 3: Verfijnde Inschatting (5 observaties)</b>',
        '<b>Stap 4: Gekalibreerd Model (7 observaties)</b>'
    ),
    horizontal_spacing=0.10,
    vertical_spacing=0.12
)

# Kernel for GP
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# Track sample points
X_samples = initial_points.copy()
y_samples = black_box_function(X_samples)

# Define iterations for each subplot
iterations = [0, 1, 3, 5]
subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for idx, (n_iter, pos) in enumerate(zip(iterations, subplot_positions)):
    row, col = pos
    
    if idx > 0:
        for i in range(iterations[idx-1], n_iter):
            if i < len(additional_points):
                X_samples = np.vstack([X_samples, additional_points[i]])
                y_samples = black_box_function(X_samples)
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=0, alpha=1e-6)
    gp.fit(X_samples, y_samples)
    y_pred, sigma = gp.predict(x_range, return_std=True)
    
    fig.add_trace(go.Scatter(x=x_range.ravel(), y=y_true.ravel(), mode='lines', name='Ware Functie', line=dict(color='#e74c3c', dash='dash', width=2), visible='legendonly', showlegend=(idx == 0), legendgroup='true'), row=row, col=col)
    fig.add_trace(go.Scatter(x=x_range.ravel(), y=y_pred, mode='lines', name='Modelvoorspelling', line=dict(color='#2ecc71', width=2), showlegend=(idx == 0), legendgroup='mean'), row=row, col=col)
    fig.add_trace(go.Scatter(x=np.concatenate([x_range.ravel(), x_range.ravel()[::-1]]), y=np.concatenate([y_pred + 1.96*sigma, (y_pred - 1.96*sigma)[::-1]]), fill='toself', fillcolor='rgba(46,204,113,0.15)', line=dict(color='rgba(255,255,255,0)'), name='95% Betrouwbaarheid', showlegend=(idx == 0), legendgroup='ci'), row=row, col=col)
    fig.add_trace(go.Scatter(x=X_samples.ravel(), y=y_samples.ravel(), mode='markers', name='Observaties', marker=dict(color='#3498db', size=10, symbol='circle', line=dict(width=2, color='white')), showlegend=(idx == 0), legendgroup='samples'), row=row, col=col)
    
    avg_uncertainty = np.mean(sigma)
    fig.add_annotation(text=f'<b>Gem. Onzekerheid: {avg_uncertainty:.2f}</b>', xref=f'x{idx+1}', yref=f'y{idx+1}', x=17.5, y=72, showarrow=False, font=dict(size=11, color='#2c3e50', family='Arial, sans-serif'), bgcolor='rgba(255,255,255,0.95)', bordercolor='rgba(44,62,80,0.3)', borderwidth=1, borderpad=5)

fig.update_layout(title={'text': '<b>De Leerreis van het Model: Van Onzekerheid naar Zekerheid</b>', 'x': 0.5, 'font': {'size': 20, 'family': 'Arial, sans-serif'}}, height=750, width=1400, showlegend=True, legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98, font=dict(size=11), bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1), plot_bgcolor='rgba(250,250,250,1)', paper_bgcolor='white', font=dict(size=11, family='Arial, sans-serif'), margin=dict(l=60, r=40, t=80, b=80))

for i in range(1, 5):
    fig.update_xaxes(title_text='Input Parameter' if i > 2 else '', title_font=dict(size=13), tickfont=dict(size=11), gridcolor='rgba(128,128,128,0.15)', showgrid=True, row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    fig.update_yaxes(title_text='Resultaat' if i % 2 == 1 else '', title_font=dict(size=13), tickfont=dict(size=11), gridcolor='rgba(128,128,128,0.15)', showgrid=True, row=(i-1)//2 + 1, col=(i-1)%2 + 1)

fig.show()

🎯 De Strategische Afweging: Exploiteren vs. Verkennen
<div style="background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px;">
<h2 style="margin: 0; font-size: 1.2em;">Waar Testen we Volgend?</h2>
<p style="font-size: 0.9em; margin: 10px 0 0 0;">De 'Strategische Beslisser' kiest zorgvuldig: gaan we voor het perfectioneren van een bekend succes, of verkleinen we onze 'blinde vlekken'?</p>
</div>

#| echo: false
#| output: true
#| label: fig-acquisition-final
#| fig-width: 12
#| fig-height: 8.5
#| fig-align: center
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def black_box_function(x):
    return np.sin(0.9 * x) * np.sinc(x * 0.2) * 50 + 25
x_range = np.linspace(0, 20, 200).reshape(-1, 1)
y_true = black_box_function(x_range)
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
X_final = np.array([[2.0], [18.0], [10.0], [5.0], [14.0], [7.5], [12.0]])
y_final = black_box_function(X_final)
gp_final = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=0, alpha=1e-6)
gp_final.fit(X_final, y_final)
y_pred_final, sigma_final = gp_final.predict(x_range, return_std=True)
mu_sample_opt = np.max(y_final)
with np.errstate(divide='ignore', invalid='ignore'):
    imp = y_pred_final.ravel() - mu_sample_opt
    Z = imp / sigma_final
    ei = imp * norm.cdf(Z) + sigma_final * norm.pdf(Z)
    ei[sigma_final == 0.0] = 0.0

fig_acq = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])

fig_acq.add_trace(go.Scatter(x=x_range.ravel(), y=y_true.ravel(), mode='lines', name='Ware Functie', line=dict(color='#e74c3c', dash='dash')), row=1, col=1)
fig_acq.add_trace(go.Scatter(x=x_range.ravel(), y=y_pred_final, mode='lines', name='Modelvoorspelling', line=dict(color='#2ecc71')), row=1, col=1)
fig_acq.add_trace(go.Scatter(x=np.concatenate([x_range.ravel(), x_range.ravel()[::-1]]), y=np.concatenate([y_pred_final + 1.96*sigma_final, (y_pred_final - 1.96*sigma_final)[::-1]]), fill='toself', fillcolor='rgba(46,204,113,0.15)', line=dict(color='rgba(255,255,255,0)'), name='95% Betrouwbaarheid'), row=1, col=1)
fig_acq.add_trace(go.Scatter(x=X_final.ravel(), y=y_final.ravel(), mode='markers', name='Observaties', marker=dict(color='#3498db', size=10, line=dict(width=2, color='white'))), row=1, col=1)
best_next_x = x_range[np.argmax(ei)]
fig_acq.add_trace(go.Scatter(x=x_range.ravel(), y=ei, mode='lines', name='Acquisitiefunctie', line=dict(color='#8e44ad', width=3), fill='tozeroy', fillcolor='rgba(142, 68, 173, 0.1)'), row=2, col=1)
fig_acq.add_vline(x=best_next_x, line_width=3, line_dash="dash", line_color="#f39c12", row=2, col=1, annotation_text="<b>Volgende Test!</b>", annotation_position="top", annotation_font_size=12)

exploit_peak_idx = np.argmax(ei[:100])
explore_peak_idx = 100 + np.argmax(ei[100:])
exploit_x = x_range[exploit_peak_idx][0]
explore_x = x_range[explore_peak_idx][0]

fig_acq.add_annotation(text="<b>EXPLOITEREN</b><br>Grote kans op hoog rendement", x=exploit_x, y=ei[exploit_peak_idx]*0.9, arrowhead=2, ax=0, ay=-35, font_size=11, row=2, col=1, bgcolor="rgba(255,255,255,0.8)")
fig_acq.add_annotation(text="<b>VERKENNEN</b><br>Gebied met grote onzekerheid", x=explore_x, y=ei[explore_peak_idx]*0.9, arrowhead=2, ax=0, ay=-35, font_size=11, row=2, col=1, bgcolor="rgba(255,255,255,0.8)")

fig_acq.update_layout(title={'text': "<b>De Strategische Keuze: Waar Testen We Volgend?</b>", 'x':0.5, 'y':0.98, 'font_size': 18}, height=750, legend=dict(x=0.99, y=0.98, xanchor='right', yanchor='top', bgcolor='rgba(255,255,255,0.7)', bordercolor='lightgrey', borderwidth=1, font_size=11), plot_bgcolor='rgba(250,250,250,1)', margin=dict(l=60, r=20, t=50, b=80))

fig_acq.update_yaxes(title_text='Resultaat', title_font_size=12, tickfont_size=10, row=1, col=1)
fig_acq.update_yaxes(title_text='Verwachte<br>Verbetering', title_font_size=12, tickfont_size=10, row=2, col=1)
fig_acq.update_xaxes(title_text='Input Parameter', title_font_size=12, tickfont_size=10, row=2, col=1)

fig_acq.show()

🔄 Het Cyclische Proces naar de Optimale Oplossing
<div style="background: linear-gradient(135deg, #ff7675 0%, #fd79a8 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px;">
<h2 style="margin: 0; font-size: 1.4em;">Vijf Stappen naar een Steeds Slimmere Keuze</h2>
</div>

<div style="display: flex; align-items: stretch; gap: 25px; margin: 30px 0 20px 0;">

<div style="flex: 1; background: #fff3cd; padding: 15px; border-radius: 10px; border: 2px solid #ffc107; text-align: center;">
<div style="background: #ffc107; color: white; border-radius: 50%; width: 32px; height: 32px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.95em;">1</div>
<h4 style="margin: 5px 0; font-size: 1.05em;">Initialiseer</h4>
<p style="margin: 0; font-size: 0.85em;">Start met enkele verkennende metingen</p>
</div>

<div style="flex: 0 0 40px; display: flex; align-items: center; justify-content: center; font-size: 1.6em; color: #495057; font-weight: bold;">→</div>

<div style="flex: 1; background: #d4edda; padding: 15px; border-radius: 10px; border: 2px solid #28a745; text-align: center;">
<div style="background: #28a745; color: white; border-radius: 50%; width: 32px; height: 32px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.95em;">2</div>
<h4 style="margin: 5px 0; font-size: 1.05em;">Modelleer</h4>
<p style="margin: 0; font-size: 0.85em;">Het model vormt een beeld van de werkelijkheid</p>
</div>

<div style="flex: 0 0 40px; display: flex; align-items: center; justify-content: center; font-size: 1.6em; color: #495057; font-weight: bold;">→</div>

<div style="flex: 1; background: #d1ecf1; padding: 15px; border-radius: 10px; border: 2px solid #17a2b8; text-align: center;">
<div style="background: #17a2b8; color: white; border-radius: 50%; width: 32px; height: 32px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.95em;">3</div>
<h4 style="margin: 5px 0; font-size: 1.05em;">Selecteer</h4>
<p style="margin: 0; font-size: 0.85em;">De 'beslisser' wijst de meest informatieve test aan</p>
</div>

<div style="flex: 0 0 40px; display: flex; align-items: center; justify-content: center; font-size: 1.6em; color: #495057; font-weight: bold;">→</div>

<div style="flex: 1; background: #f8d7da; padding: 15px; border-radius: 10px; border: 2px solid #dc3545; text-align: center;">
<div style="background: #dc3545; color: white; border-radius: 50%; width: 32px; height: 32px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.95em;">4</div>
<h4 style="margin: 5px 0; font-size: 1.05em;">Valideer</h4>
<p style="margin: 0; font-size: 0.85em;">Voer het kostbare, fysieke experiment uit</p>
</div>

<div style="flex: 0 0 40px; display: flex; align-items: center; justify-content: center; font-size: 1.6em; color: #495057; font-weight: bold;">→</div>

<div style="flex: 1; background: #e2e3e5; padding: 15px; border-radius: 10px; border: 2px solid #6c757d; text-align: center;">
<div style="background: #6c757d; color: white; border-radius: 50%; width: 32px; height: 32px; margin: 0 auto 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.95em;">5</div>
<h4 style="margin: 5px 0; font-size: 1.05em;">Optimaliseer</h4>
<p style="margin: 0; font-size: 0.85em;">Voeg het resultaat toe en herhaal de cyclus</p>
</div>

</div>

::: {style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-left: 5px solid #007bff; margin-top: 180px;"}

💡 De Kern
Het algoritme balanceert continu. Eerst het speelveld in kaart brengen (verkennen), daarna gericht de 'goudaders' aanboren (exploiteren). Dit is de sleutel tot een efficiëntie die willekeurig testen ver overtreft.
:::

🏭 De Praktijk: De Optimale DC Locatie Vinden
<p style="text-align: center; font-size: 1.05em; margin: 10px 0 20px 0; color: #666;"><strong>De Case:</strong> Bepaal de ideale DC-locatie, met een budget voor slechts enkele kostbare haalbaarheidsstudies.</p>

<div style="display: grid; grid-template-columns: 40% 30% 30%; gap: 20px; align-items: start;">

<div style="text-align: center;">
<img src="bo-optimizer.png" alt="3D Bayesian Optimization Visualization" style="width: 100%; max-width: 420px; border-radius: 8px; box-shadow: 0 3px 12px rgba(0,0,0,0.1);">
<div style="margin-top: 12px;">
<a href="bo-simulator-fast.html" target="_blank" style="display: inline-block; background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; text-decoration: none; padding: 10px 20px; border-radius: 6px; font-size: 0.95em; font-weight: bold; box-shadow: 0 3px 12px rgba(0, 184, 148, 0.3);">
🚀 Start de Simulator
</a>
<br><span style="font-size: 0.7em; color: #666;">Opent in een nieuw venster</span>
</div>
</div>

<div>
<h3 style="color: #00529b; margin: 0 0 10px 0; font-size: 1em;">🔍 Wat u gaat zien:</h3>
<div style="font-size: 0.72em; line-height: 1.3;">
<div style="margin-bottom: 6px;"><span style="color: #4682B4;">●</span> <strong>Blauwe cilinders:</strong> Uw klantlocaties</div>
<div style="margin-bottom: 6px;"><span style="color: #2ecc71;">●</span> <strong>Gekleurd oppervlak:</strong> De kostenvoorspelling van de AI</div>
<div style="margin-bottom: 6px;"><span style="color: #ff6347;">●</span> <strong>Rood raster:</strong> De (verborgen) realiteit</div>
<div><span style="color: #9370db;">●</span> <strong>Paars oppervlak:</strong> Waar de AI vervolgens wil testen</div>
</div>
</div>

<div>
<h3 style="color: #00529b; margin: 0 0 10px 0; font-size: 1em;">⚡ Experimenteer zelf:</h3>
<div style="font-size: 0.72em; line-height: 1.3;">
• Start de simulatie met <strong>'Initialize/Reset'</strong><br>
• Voer het leerproces stap voor stap uit<br>
• Pas de 'Exploration'-schuif aan<br>
• Wissel tussen voorspelling en realiteit<br>
• Ontdek hoe de AI het optimum vindt!
</div>
</div>

</div>

<div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 12px 25px; border-radius: 8px; text-align: center; margin-top: 20px;">
<strong>💡 Cruciaal Inzicht:</strong> U ziet live hoe het algoritme de strategische balans vindt tussen het verkennen van onbekend terrein en het benutten van veelbelovende locaties. Dát is de kern van efficiënt optimaliseren.
</div>

🚛 Uw Strategische Voorsprong in de Logistiek
<div style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;">
<h2 style="margin-top: 0;">Minder Experimenten, Meer Resultaat</h2>
<p style="font-size: 1.1em; margin: 0;">De formule voor efficiëntie, kostenbesparing en een versnelde innovatie.</p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 25px 0;">

<div style="background: #fff5f5; padding: 15px; border-radius: 12px; border: 2px solid #fc8181; text-align: center;">
<div style="font-size: 2.2em; margin-bottom: 10px;">🤖</div>
<h3 style="color: #e53e3e; font-size: 1.1em; margin: 0 0 8px 0;">Perfectie in Voorspelling</h3>
<p style="font-size: 0.9em; margin: 0;">Vind de optimale parameters voor uw demand forecasting- en ETA-modellen.</p>
</div>

<div style="background: #f0fff4; padding: 15px; border-radius: 12px; border: 2px solid #68d391; text-align: center;">
<div style="font-size: 2.2em; margin-bottom: 10px;">🚚</div>
<h3 style="color: #38a169; font-size: 1.1em; margin: 0 0 8px 0;">Efficiëntie op de Weg</h3>
<p style="font-size: 0.9em; margin: 0;">Kalibreer uw route-algoritmes voor minimale brandstofkosten en maximale punctualiteit.</p>
</div>

<div style="background: #fffaf0; padding: 15px; border-radius: 12px; border: 2px solid #f6ad55; text-align: center;">
<div style="font-size: 2.2em; margin-bottom: 10px;">🏭</div>
<h3 style="color: #dd6b20; font-size: 1.1em; margin: 0 0 8px 0;">Slimme Magazijnen</h3>
<p style="font-size: 0.9em; margin: 0;">Optimaliseer de configuratie van uw robotica en sorteerstrategieën via snelle, effectieve simulaties.</p>
</div>

<div style="background: #f7faff; padding: 15px; border-radius: 12px; border: 2px solid #63b3ed; text-align: center;">
<div style="font-size: 2.2em; margin-bottom: 10px;">⛓️</div>
<h3 style="color: #3182ce; font-size: 1.1em; margin: 0 0 8px 0;">Netwerkontwerp van de Toekomst</h3>
<p style="font-size: 0.9em; margin: 0;">Identificeer de ideale DC-locaties, gebaseerd op een perfecte balans tussen kosten en serviceniveau.</p>
</div>

</div>

📦 De Kern van de Zaak
::: {.center style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px;"}
In een wereld waar elk experiment telt,

is slim optimaliseren geen luxe meer. Het is een strategische noodzaak om uw concurrentiepositie te versterken.

:::

🚀 De Volgende Stap: AI die Zelf Strategieën Ontwikkelt
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
<h2 style="margin: 0; font-size: 1.2em;">Een Revolutie in Optimalisatie</h2>
<p style="font-size: 0.95em; margin: 8px 0 0 0;">We staan aan de vooravond van AI die zelf compleet nieuwe optimalisatietechnieken ontwerpt.</p>
</div>

<div style="display: grid; grid-template-columns: 1.4fr 0.9fr; gap: 30px; align-items: start;">

<div>
<h3 style="color: #6c5ce7; margin-bottom: 15px; font-size: 1.1em;">🤖 Voorbeeld: FunBO, waar AI zijn Eigen Strategieën Bouwt</h3>

<div style="display: grid; grid-template-columns: auto 1fr; gap: 12px; align-items: start; margin-bottom: 15px;">
<div style="font-size: 1.3em;">🧠</div>
<div style="font-size: 0.9em;">
<strong>Generatie:</strong> Een Large Language Model genereert code voor nieuwe beslisstrategieën.
</div>

<div style="font-size: 1.3em;">🧪</div>
<div style="font-size: 0.9em;">
<strong>Evaluatie:</strong> Een geautomatiseerd systeem test de effectiviteit van elke nieuwe strategie.
</div>

<div style="font-size: 1.3em;">🏆</div>
<div style="font-size: 0.9em;">
<strong>Evolutie:</strong> Alleen de best presterende strategieën worden behouden en doorontwikkeld.
</div>
</div>

<div style="background: #f0f8ff; padding: 12px; border-radius: 8px; border-left: 4px solid #4a90e2;">
<h4 style="color: #2c5aa0; margin: 0 0 5px 0; font-size: 0.95em;">💡 De Impact voor U</h4>
<p style="font-size: 0.85em; margin: 0;">Dit leidt tot op maat gemaakte optimalisatiestrategieën, die de standaardmethodes ver achter zich laten.</p>
</div>

<div style="margin-top: 15px; padding: 8px; background: #f9f9f9; border-radius: 6px;">
<p style="font-size: 0.65em; color: #666; margin: 0; line-height: 1.2;">
<em>Bron:</em> Aglietti et al. (2024). Funbo: Discovering acquisition functions for bayesian optimization with funsearch. <em>arXiv:2406.04824</em>.
</p>
</div>
</div>

<div style="text-align: center; background: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #dee2e6;">
<img src="fig_funbo.png" alt="FunBO Process Diagram" style="max-width: 100%; height: auto; margin-bottom: 8px;">
<p style="font-style: italic; color: #6c757d; font-size: 0.8em; margin: 0;">
<strong>Het FunBO-proces:</strong> AI genereert, test en evolueert continu om tot superieure oplossingen te komen.
</p>
</div>

</div>

🎯 De Essentie & Uw Volgende Stappen
<div style="background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px;">
<h1 style="margin-top: 0;">🧠 De Formule voor Slim Optimaliseren</h1>
<p style="font-size: 1.15em; margin: 0;">Bayesian Optimization = De Kracht van een Slimme Voorspeller + een Strategische Beslisser</p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin: 20px 0;">

<div>
<h2 style="color: #2d3436; font-size: 1.3em;">🎉 Uw Belangrijkste Inzichten</h2>
<div style="background: #dff0d8; padding: 12px; border-radius: 10px; border-left: 5px solid #5cb85c;">
<ul style="font-size: 0.85em; line-height: 1.5; margin: 0; padding-left: 15px;">
<li>Een krachtige strategie gebaseerd op <strong>voorspellen en beslissen</strong>.</li>
<li><strong>Maximale inzichten</strong> met een minimaal aantal dure tests.</li>
<li>De cruciale, strategische balans tussen <strong>verkennen en exploiteren</strong>.</li>
<li>Directe impact op <strong>kosten, efficiëntie en innovatiekracht</strong>.</li>
</ul>
</div>
</div>

<div>
<h2 style="color: #2d3436; font-size: 1.3em;">🚀 Uw Concrete Volgende Stappen</h2>
<div style="background: #d9edf7; padding: 12px; border-radius: 10px; border-left: 5px solid #5bc0de;">
<ol style="font-size: 0.85em; line-height: 1.5; margin: 0; padding-left: 15px;">
<li><strong>Identificeer:</strong> Waar in uw proces zitten de kostbare 'trial-and-error'-cycli?</li>
<li><strong>Activeer:</strong> Ga het gesprek aan met uw data- en R&D-teams.</li>
<li><strong>Experimenteer:</strong> Start een pilot met één afgebakend, complex probleem.</li>
<li><strong>Valideer:</strong> Meet de resultaten en bouw de business case.</li>
</ol>
</div>
</div>

</div>

🎤 Vragen & Discussie
::: {style="background: #f8f9fa; padding: 30px; border-radius: 15px; border: 2px solid #28a745;"}
🎤

Bent u er klaar voor om te ontdekken waar Bayesian Optimization uw logistieke operatie kan versterken?

Laten we de discussie openen en uw specifieke uitdagingen en kansen verkennen.

:::

🙏 Hartelijk Dank {.center}
::: {.center style="padding: 40px; background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; border-radius: 15px;"}
Contactgegevens HAN Lectoraat Logistiek en Allianties | Karen.Engelvaart@han.nl

Deze presentatie bevat interactieve demo's over Bayesian Optimization
:::