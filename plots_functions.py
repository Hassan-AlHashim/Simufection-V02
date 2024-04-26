import numpy as np
import pandas as pd
import plotly.graph_objects as go  
from plotly.subplots import make_subplots 

import os  
import json 


def setup_simulation_parameters(alpha, beta, kappa, location_data):

    p = {}
    original_coords = np.array([
    [29.7604, -95.3698],  # Houston, Texas
    [33.4484, -112.0740],  # Phoenix, Arizona
    [39.7392, -104.9903],  # Denver, Colorado
    [32.7157, -117.1611],  # San Diego, California
])
    
    additional_coords = np.array([
    [
        loc.get('x', loc.get('X Coordinate')),  
        loc.get('y', loc.get('Y Coordinate')) 
    ] 
    for loc in location_data
])

    updated_coords = additional_coords if additional_coords.size > 0 else original_coords
    new_N = updated_coords.shape[0]

    populations = [
    loc.get('population', loc.get('Population')) 
    for loc in location_data
]

    infection_sources = [
        loc.get('infection_source', loc.get('Infection Source'))  
        for loc in location_data
    ]

    vaccination_rates = [
        loc.get('vaccination_rate', loc.get('Vaccination Rate'))  
        for loc in location_data
    ]

    p['mode'] = 'exp'
    p['coords'] = updated_coords
    p['N'] = new_N
    p['P'] = np.ones([new_N, new_N])

    if location_data:
        node_names = [loc.get('name', f'Node {i}') for i, loc in enumerate(location_data)]
    else:
        node_names = [f'Node {i}' for i in range(p['N'])]
    p['node_names'] = node_names

    #print("Location Data:", location_data)

    mobility_matrix = []
    mobility_matrix = [
    [0.7, 0.3, 1] if loc.get('mobility', loc.get('Mobility')) == 'Yes' else [0, 0, 0]
    for loc in location_data
    ]

    if mobility_matrix: 
        print("Using new mobility values")
        p['M'] = np.array(mobility_matrix)
    else: 
        print("Using default mobility values")
        p['M'] = np.tile(np.array([0.7, 0.3, 1])[np.newaxis, :], [new_N, 1])
    
    for i, mobility_vals in enumerate(mobility_matrix):
        if all(val == 0 for val in mobility_vals):
            p['P'][i, :] = 0
            p['P'][:, i] = 0
        #print("Mobility Matrix p['M']:", p['M'])
    
    beta= 0.5*beta

    p['alpha'] = np.full(new_N, float(alpha))
    p['beta'] = np.full(new_N, float(beta))
    p['kappa'] = np.full(new_N, float(kappa))
    p['mu'] = np.tile(0.00001 * np.array([1.00, 1.00, 1.00])[np.newaxis, :], [new_N, 1])
    p['gamma'] = np.full(new_N, 0.1)
    p['zeta'] = np.full(new_N, 0.1)
    p['u0'] = np.full(new_N, 0)
    p['q0'] = np.array([10] + [0]*(new_N-1)) if not infection_sources else np.array(infection_sources)
    p['v'] = np.array([1]*new_N) if not vaccination_rates else np.array(vaccination_rates)
    p['S0'] = np.random.randint(10000, 30000, size=new_N) if not populations else np.array(populations)
    return p

def setup_map_figure(params, timesteps, state_trajectories, tau_values):
    map_fig = go.Figure()

    step_size = 5  # You can adjust this value to control frame generation
    frames = []
    for idx, (t, state, tau) in enumerate(zip(timesteps, state_trajectories, tau_values)):
        if idx % step_size == 0:  # Only build frames at each step
            frame_traces = build_map_traces(state, params['coords'], tau, map_type='mapbox')
            frames.append(go.Frame(data=frame_traces, name=str(idx)))

    slider_steps = setup_slider_controls(frames, timesteps)
    map_fig.frames = frames
    if frames:
        map_fig.add_traces(frames[0].data)

    map_fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=37.04358, lon=-97.23412), zoom=3),
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        sliders=[slider_steps],
        updatemenus=[{
        'type': 'buttons',
        'buttons': [
            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]},
            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }],
        legend=dict(x=0, y=1, orientation='h', bgcolor='rgba(0,0,0,0.7)'),
        font=dict(family="Arial, sans-serif", size=12, color="white"),
        paper_bgcolor='black',
        plot_bgcolor='black'
    )

    print("Map figure setup with frames:", len(frames))

    return map_fig

def setup_3dmap_figure(params, timesteps, state_trajectories, tau_values):
    map_fig3d = go.Figure()

    step_size = 5
    frames = []
    for idx, (t, state, tau) in enumerate(zip(timesteps, state_trajectories, tau_values)):
        if idx % step_size == 0:
            frame_traces = build_map_traces(state, params['coords'], tau, map_type='geo')
            frames.append(go.Frame(data=frame_traces, name=str(idx)))

    slider_steps = setup_slider_controls(frames, timesteps)
    map_fig3d.frames = frames
    if frames:
        map_fig3d.add_traces(frames[0].data)

    map_fig3d.update_layout(
        geo=dict(
            projection_type="orthographic",
            showland=True,
            landcolor="rgb(190, 173, 150)",  
            showocean=True,
            oceancolor="rgb(29,162,216)",
            showlakes=True,
            lakecolor="rgb(127,205,255)",  
            bgcolor='rgb(10,10,10)',  
            showcountries=True,  
            countrycolor="rgb(0, 4, 8)", 
            showsubunits=True,  
            subunitcolor="rgb(0, 4, 8)",
            resolution=110  
        ),
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        sliders=[slider_steps],
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        legend=dict(x=0, y=1, orientation='h', bgcolor='rgba(0,0,0,0.7)'),
        font=dict(family="Arial, sans-serif", size=12, color="white"),
        paper_bgcolor='black',  
        plot_bgcolor='black'  
    )

    print("3-D Map figure setup with frames:", len(frames))

    return map_fig3d


def build_map_traces(state, coords, tau, map_type='mapbox'):
    S, I, R = state[:, 0], state[:, 1], state[:, 2]
    Total = S + I + R
    
    max_total = Total.max()
    if max_total == 0:
        max_total = 1  # Avoid division by zero

    T_normalized = Total / max_total

    S_scale = np.where(Total > 0, S / Total, 0)
    I_scale = np.where(Total > 0, I / Total, 0)
    R_scale = np.where(Total > 0, R / Total, 0)

    max_size = 50
    min_size = 2

    S_sizes = np.clip(T_normalized * max_size * S_scale, min_size, max_size)
    I_sizes = np.clip(T_normalized * max_size * I_scale + S_sizes, min_size, max_size)
    R_sizes = np.clip(T_normalized * max_size * R_scale + I_sizes, min_size, max_size)

    scatter_class = go.Scattermapbox if map_type == 'mapbox' else go.Scattergeo
    line_color = 'yellow' if map_type == 'mapbox' else 'black'  

    traces = [
        scatter_class(
            lat=coords[:, 0], lon=coords[:, 1], mode='markers',
            marker=dict(size=[s+2 for s in R_sizes], color='black'),
            name='Recovered_border',
            showlegend=False
        ),
        scatter_class(
            lat=coords[:, 0], lon=coords[:, 1], mode='markers',
            marker=dict(size=R_sizes, color='green', opacity=0.7), name='Recovered',
            hoverinfo='text',
            hovertext=['Recovered: ' + '{:0.0f}'.format(r) for r in R]
        ),
        scatter_class(
            lat=coords[:, 0], lon=coords[:, 1], mode='markers',
            marker=dict(size=[s+2 for s in I_sizes], color='black'),
            name='Infected_border',
            showlegend=False
        ),
        scatter_class(
            lat=coords[:, 0], lon=coords[:, 1], mode='markers',
            marker=dict(size=I_sizes, color='red', opacity=0.7), name='Infected',
            hoverinfo='text',
            hovertext=['Infected: ' + '{:0.0f}'.format(inff) for inff in I]
        ),
        scatter_class(
            lat=coords[:, 0], lon=coords[:, 1], mode='markers',
            marker=dict(size=[s+2 for s in S_sizes], color='black'),
            name='Susceptible_border',
            showlegend=False
        ),
        scatter_class(
            lat=coords[:, 0], lon=coords[:, 1], mode='markers',
            marker=dict(size=S_sizes, color='blue', opacity=0.7), name='Susceptible',
            hoverinfo='text',
            hovertext=['Susceptible: ' + '{:0.0f}'.format(s) for s in S]
        )
    ]

    line_base_width = 1 
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            tau_sum = tau[i, j].sum()
            if tau_sum > 0:
                width = max(line_base_width, 2 * tau_sum)  
                traces.append(scatter_class(
                    lat=[coords[i, 0], coords[j, 0]],
                    lon=[coords[i, 1], coords[j, 1]],
                    mode='lines',
                    line=dict(width=width, color=line_color),
                    hoverinfo='none', showlegend=False
                ))

    return traces

def setup_slider_controls(frames, timesteps, duration=100):
    steps = []
    for frame in frames:
        step_time = int(timesteps[int(frame.name)])
        step = {
            "args": [
                [frame.name],
                {"frame": {"duration": duration, "redraw": True}, "mode": "immediate", "transition": {"duration": duration}}
            ],
            "label": str(step_time),
            "method": "animate"
        }
        steps.append(step)
    return {
        'steps': steps,
        'transition': {'duration': 300},
        'x': 0.1,
        'y': 0,
        'currentvalue': {
            'visible': True,
            'prefix': 'Time: ',
            'xanchor': 'right'
        },
        'pad': {'b': 10, 't': 10},
        'len': 0.9,
        'xanchor': 'left',
        'yanchor': 'top'
    }

def setup_stats_figure(params, state_trajectories, timesteps):
    num_nodes = params['N']
    num_states = state_trajectories.shape[2]
    state_dict = {0: 'S', 1: 'I', 2: 'R'}

    node_names = params.get('node_names', [f'Node {i}' for i in range(num_nodes)])

    stats_fig = go.Figure()

    for node in range(num_nodes):
        for state in range(num_states):
            stats_fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='lines+markers',
                name=f'{node_names[node]}, State {state_dict[state]}',
                visible=True if node == 0 else 'legendonly'
            ))

    frames = []
    step_size = 5
    for idx, t in enumerate(timesteps):
        if idx % step_size == 0:
            frame_traces = []
            for node in range(num_nodes):
                for state in range(num_states):
                    x_data = timesteps[:idx + 1]
                    y_data = state_trajectories[:idx + 1, node, state]
                    frame_traces.append(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines+markers',
                        name=f'{node_names[node]}, State {state_dict[state]}'
                    ))
            frames.append(go.Frame(data=frame_traces, name=str(idx)))

    stats_fig.frames = frames
    print("Stat figure setup with frames:", len(frames))

    buttons = []
    for node in range(num_nodes):
        visibility = [(i == node) for i in range(num_nodes) for _ in range(num_states)]
        buttons.append(dict(
            label=node_names[node], 
            method='update',
            args=[{'visible': visibility}]
        ))

    slider_settings = setup_slider_controls(frames, timesteps)
    
    stats_fig.update_layout(
        updatemenus=[
            {
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'xanchor': 'right',
                'y': 1.2,
                'yanchor': 'top'
            },
            {
                'type': 'buttons',
                'buttons': [
                    {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]},
                    {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'showactive': True,
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }
        ],
        sliders=[slider_settings],
        paper_bgcolor='rgba(0,0,0,0.8)',
        plot_bgcolor='rgba(0,0,0,0.8)',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='gray',
            linecolor='white', 
            range=[0, max(timesteps)]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gray',
            linecolor='white'
        )
    )

    return stats_fig
