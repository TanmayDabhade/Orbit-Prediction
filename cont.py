import numpy as np
import plotly.graph_objects as go
from math import pi
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import time
import threading

# Constants
EARTH_RADIUS = 6371  # Earth radius in kilometers

# Dummy satellite position function (Replace with actual API call)
def get_sat_position():
    return np.random.uniform(-7000, 7000, 3)  # Simulated ISS position data

# Function to convert Keplerian elements to Cartesian coordinates (stub)
def kepler_to_cartesian(semi_major_axis, eccentricity, inclination, raan, arg_periapsis, anomaly):
    # Replace with actual conversion logic
    return np.array([semi_major_axis * np.cos(anomaly), 
                     semi_major_axis * np.sin(anomaly), 
                     semi_major_axis * np.sin(inclination)]), None

# Orbit Calculation
def calculate_orbit(semi_major_axis, eccentricity, inclination, raan, arg_periapsis):
    true_anomalies = np.linspace(0, 2 * pi, 100)
    orbit = np.array([kepler_to_cartesian(semi_major_axis, eccentricity, inclination,
                                          raan, arg_periapsis, anomaly)[0]
                      for anomaly in true_anomalies])
    return orbit

# Dash app setup
app = Dash(__name__)

# Initial orbit data
orbit = calculate_orbit(7000, 0.01, 0.5, 0, 0)

# Initial Plotly figure
fig = go.Figure()

# Plot predicted orbit path
fig.add_trace(go.Scatter3d(
    x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2], mode='lines', name='Orbit Path'
))

# Plot Earth
u = np.linspace(0, 2 * pi, 50)
v = np.linspace(0, pi, 50)
x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', name='Earth'))

# Add ISS marker trace
iss_trace = go.Scatter3d(
    x=[], y=[], z=[], mode='markers+lines', marker=dict(size=5, color='red'), name='ISS'
)
fig.add_trace(iss_trace)

# Layout settings
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (km)', range=[-10000, 10000]),
        yaxis=dict(title='Y (km)', range=[-10000, 10000]),
        zaxis=dict(title='Z (km)', range=[-10000, 10000])
    ),
    title='3D Orbital Prediction and Real-Time ISS Tracking'
)

# App layout with graph component
app.layout = html.Div([
    dcc.Graph(id='live-orbit-graph', figure=fig),
    dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0)  # 5-second interval
])

# Callback to update ISS position
@app.callback(
    Output('live-orbit-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_iss_position(n):
    position = get_sat_position()  # Get current ISS position

    # Update ISS trace with new data
    iss_trace.x += (position[0],)
    iss_trace.y += (position[1],)
    iss_trace.z += (position[2],)

    # Update figure with new ISS trace
    fig.update_traces(selector=dict(name='ISS'), 
                      x=iss_trace.x, y=iss_trace.y, z=iss_trace.z)
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
