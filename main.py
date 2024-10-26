import numpy as np
import plotly.graph_objs as go
import time
from math import sqrt, cos, sin, radians, pi, atan2
from skyfield.api import load, wgs84

# Constants
MU = 398600
EARTH_RADIUS = 6371

# Keplerian to Cartesian conversion
def kepler_to_cartesian(semi_major_axis, eccentricity, inclination, raan, arg_periapsis, true_anomaly):
    p = semi_major_axis * (1 - eccentricity ** 2)
    r = p / (1 + eccentricity * cos(true_anomaly))

    x_orb = r * cos(true_anomaly)
    y_orb = r * sin(true_anomaly)

    v_x_orb = -sqrt(MU / p) * sin(true_anomaly)
    v_y_orb = sqrt(MU / p) * (eccentricity + cos(true_anomaly))

    incl = radians(inclination)
    raan = radians(raan)
    arg_periapsis = radians(arg_periapsis)

    R1 = np.array([[cos(raan), -sin(raan), 0], [sin(raan), cos(raan), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, cos(incl), -sin(incl)], [0, sin(incl), cos(incl)]])
    R3 = np.array([[cos(arg_periapsis), -sin(arg_periapsis), 0], 
                   [sin(arg_periapsis), cos(arg_periapsis), 0], [0, 0, 1]])

    rotation_matrix = R1 @ R2 @ R3
    r_vec = rotation_matrix @ np.array([x_orb, y_orb, 0])
    v_vec = rotation_matrix @ np.array([v_x_orb, v_y_orb, 0])

    return r_vec, v_vec

# Predict position with perturbations
def predict_position_with_perturbations(semi_major_axis, eccentricity, mean_anomaly, time, period):
    mean_motion = 2 * pi / period
    M = mean_anomaly + mean_motion * time
    E = solve_kepler(M, eccentricity)
    true_anomaly = 2 * atan2(sqrt(1 + eccentricity) * sin(E / 2), 
                             sqrt(1 - eccentricity) * cos(E / 2))

    position, _ = kepler_to_cartesian(semi_major_axis, eccentricity, 0, 0, 0, true_anomaly)
    return apply_perturbations(position)

# Solve Kepler's equation
def solve_kepler(M, e, tol=1e-6):
    E = M
    while True:
        delta = E - e * sin(E) - M
        if abs(delta) < tol:
            break
        E -= delta / (1 - e * cos(E))
    return E

# Apply perturbations (Sun/Moon effects)
def apply_perturbations(position, perturbation_factor=1e-3):
    sun_effect = perturbation_factor * np.array([1, 0, 0])
    moon_effect = perturbation_factor * np.array([0, 1, 0])
    return position + sun_effect + moon_effect


def get_orbital_parameters(satellite_name='ISS (ZARYA)'):
    # Load the TLE data from Celestrak
    stations_url = 'https://celestrak.com/NORAD/elements/stations.txt'
    satellites = load.tle_file(stations_url)
    satellite = {sat.name: sat for sat in satellites}[satellite_name]
    inclination = satellite.model.inclo
    raan = satellite.model.nodeo
    eccentricity = satellite.model.ecco
    arg_periapsis = satellite.model.argpo
    mean_anomaly = satellite.model.mo
    mean_motion = satellite.model.no_kozai
    MU = 398600.4418
    mean_motion_rad = mean_motion * (2 * pi) / (24 * 3600)
    semi_major_axis = (MU / mean_motion_rad**2) ** (1 / 3)



    return (semi_major_axis, eccentricity, inclination, raan, arg_periapsis, mean_anomaly)



# Fetch real-time TLE data for ISS
def get_sat_position():
    stations_url = 'https://celestrak.com/NORAD/elements/stations.txt'
    satellites = load.tle_file(stations_url)
    satellite = {sat.name: sat for sat in satellites}['ISS (ZARYA)']
    ts = load.timescale()
    t = ts.now()
    position = satellite.at(t).position.km
    return position

def get_satellite_coordinates(satellite_name):
    try:
        stations_url = 'https://celestrak.com/NORAD/elements/stations.txt'
        satellites = load.tle_file(stations_url)
        satellite = {sat.name: sat for sat in satellites}[satellite_name]
        ts = load.timescale()
        t = ts.now()
        geocentric = satellite.at(t)
        position_km = geocentric.position.km
        subpoint = wgs84.subpoint(geocentric)
        latitude = subpoint.latitude.degrees
        longitude = subpoint.longitude.degrees
        altitude = subpoint.elevation.km

        print(f"{satellite_name} Cartesian Coordinates (km): X={position_km[0]:.2f}, Y={position_km[1]:.2f}, Z={position_km[2]:.2f}")
        print(f"{satellite_name} Geographic Coordinates: Latitude={latitude:.2f}°, Longitude={longitude:.2f}°, Altitude={altitude:.2f} km")

        return position_km, (latitude, longitude, altitude)
    
    except KeyError:
        print(f"Satellite '{satellite_name}' not found.")
        return None, None




# 3D plot with real-time ISS updates and orbital prediction
def plot_3d_orbit_and_track_iss(semi_major_axis, eccentricity, inclination, raan, arg_periapsis):
    true_anomalies = np.linspace(0, 2 * pi, 100)
    orbit = np.array([kepler_to_cartesian(semi_major_axis, eccentricity, inclination, 
                                          raan, arg_periapsis, anomaly)[0] 
                      for anomaly in true_anomalies])
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2], 
        mode='lines', name='Orbit Path'
    ))
    u = np.linspace(0, 2 * pi, 50)
    v = np.linspace(0, pi, 50)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', name='Earth'))
    iss_trace = go.Scatter3d(
        x=[], y=[], z=[], mode='markers+lines', marker=dict(size=5, color='red'), name='ISS'
    )
    fig.add_trace(iss_trace)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (km)', range=[-10000, 10000]),
            yaxis=dict(title='Y (km)', range=[-10000, 10000]),
            zaxis=dict(title='Z (km)', range=[-10000, 10000])
        ),
        title='3D Orbital Prediction and Real-Time ISS Tracking'
    )
    fig.show()
    while True:
        try:
            position = get_sat_position()
            iss_trace.x = np.append(iss_trace.x, position[0])
            iss_trace.y = np.append(iss_trace.y, position[1])
            iss_trace.z = np.append(iss_trace.z, position[2])
            fig.update_traces(selector=dict(name='ISS'), 
                              x=iss_trace.x, y=iss_trace.y, z=iss_trace.z)

            fig.update()

            sat_coordinates = get_satellite_coordinates('ISS (ZARYA)') 

            print(f"Updated ISS Position: X={position[0]:.2f} km, Y={position[1]:.2f} km, Z={position[2]:.2f} km")


            time.sleep(5)
        except KeyboardInterrupt:
            print("Tracking stopped.")
            break

params = get_orbital_parameters("ISS (ZARYA)")


plot_3d_orbit_and_track_iss(params[0], params[1], params[2], params[3], params[4])  # Start the 3D plot with tracking
