import dash
from dash import dcc, html, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
import base64
import io
import plotly.express as px
import pandas as pd
import dash_ag_grid as dag
from dash import dash_table
import json
from dash.exceptions import PreventUpdate
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai
from openai import OpenAI
import os

with open('openai_key.json', 'r') as file:
    data = json.load(file)
api_key = data['API_KEY']
#print(api_key)
client = OpenAI(api_key=api_key)

if not nltk.download('stopwords'):
    nltk.download('stopwords')

if not nltk.download('punkt'):
    nltk.download('punkt')


np.random.seed(42)


css_style = """
body {
    font-family: Arial, sans-serif !important;
}

/* DataTable Styling for Dark Theme */
.dash-spreadsheet, .dash-spreadsheet-container, .dash-header, .dash-cell div {
    color: #FFFFFF; /* Light text color for better readability */
}

.dash-spreadsheet-container .Select-menu-outer {
    background-color: #2C3E50; /* Dark background for dropdowns */
}

.dash-spreadsheet tr th, .dash-spreadsheet tr td {
    background-color: #34495E; /* Dark background color for table cells */
    border: 1px solid #2C3E50; /* Slightly darker border color */
}

.dash-spreadsheet tr th {
    background-color: #2C3E50; /* Slightly darker background color for headers */
}

/* Adjusting filter input box for dark theme */
.dash-filter input {
    background-color: #2C3E50;
    color: #FFF;
}

/* Adjusting pagination for dark theme */
.dash-pagination-container .pagination {
    background-color: #34495E;
    color: #FFF;
}

.dash-pagination-container .pagination .page-item.active .page-link {
    background-color: #18BC9C;
    border-color: #18BC9C;
}

.dash-pagination-container .pagination .page-item.disabled .page-link {
    color: #777;
}

"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, css_style], suppress_callback_exceptions=True)

initial_figure = go.Figure(
    go.Scattermapbox(
        lat=['37.04358'],  
        lon=['-97.23412'],  
        mode='markers'
    )
)

initial_figure.update_layout(
    mapbox_style="light",  
    mapbox=dict(
        center=dict(lat=37.04358, lon=-97.23412), 
        zoom=2  
    ),
    margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
)

is_simulation_aborted = False

@app.callback(
    Output('abort-flag-store', 'data'),
    [Input('abort-button', 'n_clicks')],
    prevent_initial_call=True
)
def abort_simulation(n_clicks):
    return {'is_aborted': True}

@app.callback(
    [
        Output('reset-upload-store', 'data'),
        Output('input-location-name', 'value'),
        Output('input-x-coordinate', 'value'),
        Output('input-y-coordinate', 'value'),
        Output('input-population', 'value'),
        Output('input-infection-source', 'value'),
        Output('input-vaccination-rate', 'value'),
        Output('input-mobility-toggle', 'value'),
    ],
    Input('clear-locations-btn', 'n_clicks'),
    State('reset-upload-store', 'data'),
    prevent_initial_call=True
)
def trigger_reset(n_clicks, reset_counter):
    return reset_counter + 1, '', None, None, None, 50, 0.5, False

@app.callback(
    Output('upload-data-container', 'children'),  
    Input('reset-upload-store', 'data')
)
def update_upload_component(reset_counter):
    return dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Data', id='upload-button', className='btn btn-info', n_clicks=0),
        multiple=True
    )

@app.callback(
    [Output('locations-table', 'data'),
     Output('status-message', 'children')],
    [Input('location-data-store', 'data')]
)
def update_locations_and_message(stored_data):
    if not stored_data:
        return [], "No locations added yet."
    
    message = "Locations updated successfully."
    return stored_data, message
@app.callback(
    Output('simulation-trigger', 'data'),
    Input('locations-table', 'data'),
    State('simulation-trigger', 'data')
)
def update_simulation_trigger(table_data, trigger_data):
    new_version = trigger_data['version'] + 1
    return {'version': new_version}

location_data = []

def create_initial_map_layout():
    fig = go.Figure()

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=37.04358, lon=-97.23412), zoom=3),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        paper_bgcolor="lightgray",
        plot_bgcolor="lightgray",
    )
    return fig

def standardize_data_keys(data):
    """
    Standardize the keys of dictionaries in the list `data` to match the expected
    keys by the DataTable, handling different naming conventions and case sensitivity.
    """
    standardized_data = []
    key_mapping = {
        'name': ['Name', 'location', 'Location', 'place', 'Place', 'locations', 'Locations', 'places', 'Places'],
        'x': ['X Coordinate', 'x coordinate', 'X', 'x'],
        'y': ['Y Coordinate', 'y coordinate', 'Y', 'y'],
        'population': ['Population', 'population'],
        'infection_source': ['Infection Source', 'infection source', 'InfectionSource'],
        'vaccination_rate': ['Vaccination Rate', 'vaccination rate', 'VaccinationRate'],
        'mobility': ['Mobility', 'mobility']
    }
    
    inverted_mapping = {v.lower(): k for k, vals in key_mapping.items() for v in vals}
    
    for row in data:
        standardized_row = {}
        for key, value in row.items():
            standardized_key = inverted_mapping.get(key.lower(), key)  # Default to original key if no mapping found
            standardized_row[standardized_key] = value
        standardized_data.append(standardized_row)
    
    return standardized_data

def add_detailed_descriptions(df):
    previous_row = None
    previous_node = None
    
    def get_detailed_description(current_row, previous_row):
        if previous_row is None or current_row['Node'] != previous_row['Node']:
            return "Initial data point for node."
        
        desc = []
        
        if current_row['Infected'] > previous_row['Infected']:
            if current_row['Infected'] > previous_row['Recovered']:
                desc.append("Infection rising sharply.")
            else:
                desc.append("Infection increasing but recovery is higher.")
        elif current_row['Infected'] < previous_row['Infected']:
            desc.append("Infection declining.")
        
        if current_row['Recovered'] > previous_row['Recovered']:
            desc.append("Recovery numbers improving.")
        
        if current_row['Susceptible'] < previous_row['Susceptible']:
            desc.append("Susceptible population decreasing.")
        
        if not desc: 
            return "Stable condition with no significant changes from previous timestep."
        
        return ' '.join(desc)
    
    descriptions = []
    for index, row in df.iterrows():
        description = get_detailed_description(row, previous_row if previous_node == row['Node'] else None)
        descriptions.append(description)
        previous_row = row
        previous_node = row['Node']
    
    df['Description'] = descriptions
    return df

@app.callback(
    Output('location-data-store', 'data'),
    [Input('upload-data', 'contents'),
     Input('add-location-btn', 'n_clicks'),
     Input('clear-locations-btn', 'n_clicks'),
     Input('locations-table', 'data_timestamp')],
    [State('upload-data', 'filename'),
     State('input-location-name', 'value'),
     State('input-x-coordinate', 'value'),
     State('input-y-coordinate', 'value'),
     State('input-population', 'value'),
     State('input-infection-source', 'value'),
     State('input-vaccination-rate', 'value'),
     State('input-mobility-toggle', 'value'),
     State('location-data-store', 'data'),
     State('locations-table', 'data')]
)
def unified_data_handler(contents, add_clicks, clear_clicks, table_timestamp,
                         filenames, name, x, y, population, infection_source,
                         vaccination_rate, mobility, existing_data, table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'upload-data' and contents:
        combined_data = existing_data[:]
        for content, filename in zip(contents, filenames):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            if 'xlsx' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
                new_data = standardize_data_keys(df.to_dict('records'))
                combined_data += new_data
        return combined_data

    elif triggered_id == 'add-location-btn' and name:
        mobility_text = "Yes" if mobility else "No"
        
        new_entry = standardize_data_keys([{
            'Name': name,
            'X Coordinate': x,
            'Y Coordinate': y,
            'Population': population,
            'Infection Source': infection_source,
            'Vaccination Rate': vaccination_rate,
            'Mobility': mobility_text 
        }])[0]
        
        updated_data = existing_data[:]
        updated_data.append(new_entry)
        return updated_data

    elif triggered_id == 'clear-locations-btn':
        return []

    return table_data or existing_data

@app.callback(
    Output('simulation-plot', 'figure'),
    [Input('run-button', 'n_clicks')],
    [
        State('abort-flag-store', 'data'), 
        State('alpha', 'value'), 
        State('beta', 'value'),
        State('kappa', 'value'),
        State('duration-slider', 'value'), 
        State('time-step-slider', 'value'), 
        State('location-data-store', 'data')
    ],
    #prevent_initial_call=True
)

@app.callback(
    [Output('map-plot', 'figure'), Output('3d-map-plot', 'figure'), Output('statistics-plot', 'figure')],
    Input('run-button', 'n_clicks'),
    State('abort-flag-store', 'data'),
    State('alpha', 'value'), 
    State('beta', 'value'),
    State('kappa', 'value'), 
    State('duration-slider', 'value'), 
    State('time-step-slider', 'value'), 
    State('location-data-store', 'data')
)
def update_simulation(n_clicks, abort_flag_data, alpha, beta, kappa, duration, time_step, location_data):
    if n_clicks is None:
        return go.Figure(), go.Figure(), go.Figure()

    if abort_flag_data.get('is_aborted', False):
        return go.Figure(), go.Figure(), go.Figure()

    sim_params = setup_simulation_parameters(alpha, beta, kappa, location_data)
    sim = Simulator(sim_params)
    tspan = (0, duration if duration is not None else 350)
    dt = time_step if time_step is not None else 0.5
    timesteps, state_trajectories, tau_values = sim.forward_euler(tspan, dt, return_tau=True)

    results = []
    node_names = sim_params['node_names']
    for node_index, node_name in enumerate(node_names):
        for time_index, time in enumerate(timesteps):
            results.append({
                'Time': time,
                'Node': node_name,
                'Susceptible': state_trajectories[time_index, node_index, 0],
                'Infected': state_trajectories[time_index, node_index, 1],
                'Recovered': state_trajectories[time_index, node_index, 2]
            })
    
    df = pd.DataFrame(results)
    df.to_excel('raw_output.xlsx', index=False)

    df_descriptions = add_detailed_descriptions(df)

    df_descriptions.to_excel('output_with_descriptions.xlsx', index=False)

    map_fig = setup_map_figure(sim_params, timesteps, state_trajectories, tau_values)
    map_fig3d = setup_3dmap_figure(sim_params, timesteps, state_trajectories, tau_values)
    stats_fig = setup_stats_figure(sim_params, state_trajectories, timesteps)

    return map_fig, map_fig3d, stats_fig

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

@app.callback(
    Output('duration-value', 'children'),
    [Input('duration-slider', 'value')]
    )
def update_duration_display(value):
        return f"{value} days"

@app.callback(
        Output('time-step-value', 'children'),
        [Input('time-step-slider', 'value')]
    )
def update_time_step_display(value):
        return f"Time step: {value}"
    
@app.callback(
    Output('infection-source-value', 'children'),
    [Input('input-infection-source', 'value')]
)
def update_infection_source_display(value):
    return f"Infection source: {value}"

@app.callback(
    Output('vaccination-rate-value', 'children'),
    [Input('input-vaccination-rate', 'value')]
)
def update_vaccination_rate_display(value):
    return f"Vaccination rate: {value:.2f}"
    
@app.callback(
    [Output('alpha', 'value'), Output('beta', 'value'),
     Output('kappa', 'value'),
     Output('alpha-value', 'children'), Output('beta-value', 'children'),
     Output('kappa-value', 'children')],
    [Input('preset-1', 'n_clicks'), Input('preset-2', 'n_clicks'),
     Input('alpha', 'value'), Input('beta', 'value'),
     Input('kappa', 'value')],
    prevent_initial_call=True
)
def update_values_and_displays(preset1, preset2, alpha, beta, kappa):
    ctx = callback_context
    if not ctx.triggered:
        triggered_id = 'No clicks yet'
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'preset-1':
        return 0.35, 0.05, 0.03, "0.35", "0.05", "0.03"
    elif triggered_id == 'preset-2':
        return 0.45, 0.08, 0.06, "0.45", "0.08", "0.06"
    else:
        return alpha, beta, kappa, f"{alpha:.2f}", f"{beta:.2f}", f"{kappa:.2f}"
        
app.layout = dbc.Container(
    [
        dcc.Store(id='abort-flag-store', data={'is_aborted': False}),
        dcc.Store(id='location-data-store', storage_type='session', data=[]),
        dcc.Store(id='upload-key-store', data={'key': 'upload-data-1'}),
        dcc.Store(id='reset-upload-store', data=0),
        dcc.Store(id='simulation-trigger', data={'version': 0}),
        dcc.Store(id='conversation-history', data=[]),
        dbc.Row(
            dbc.Col(
                html.H4(
                    "Simufection (Computational Engineering & Data Analytics)",
                    className="text-white bg-primary p-2 mb-2 text-center"
                ),
                width=12
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                html.H5("Input Parameters", className="card-title"),
                                html.Div(
                                    [
                                        html.Label("α (Infection Rate)", className="mb-1"),
                                        dcc.Slider(
                                            id='alpha',
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.42,
                                             marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                        ),
                                        html.Div(id='alpha-value', children="α: 0.42"),
                                    ],
                                    className="mb-4"
                                ),
                                html.Div(
                                    [
                                        html.Label("β (Recovery Rate)", className="mb-1"),
                                        dcc.Slider(
                                            id='beta',
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.07,
                                             marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                        ),
                                        html.Div(id='beta-value', children="β: 0.07"),
                                    ],
                                    className="mb-4"
                                ),
                                html.Div(
                                    [
                                        html.Label("κ (Mortality Rate)", className="mb-1"),
                                        dcc.Slider(
                                            id='kappa',
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.05,
                                             marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                        ),
                                        html.Div(id='kappa-value', children="κ: 0.05"),
                                    ],
                                    className="mb-4"
                                ),
                            ],
                            body=True,
                        ),
                        dbc.Card(
                            [
                                html.H5("Dynamic Parameters", className="card-title"),
                                html.Div(
                                    [
                                        html.Label("Duration (days)", className="mb-1"),
                                        dcc.Slider(
                                            id='duration-slider',
                                            min=0,
                                            max=2000,
                                            step=1,
                                            value=350,
                                            marks={i: str(i) for i in range(0, 2001, 500)},
                                        ),
                                        html.Div(id='duration-value', children="Duration: 350 days"),
                                    ],
                                    className="mb-4"
                                ),
                                html.Div(
                                    [
                                        html.Label("Time Step", className="mb-1"),
                                        dcc.Slider(
                                            id='time-step-slider',
                                            min=0,
                                            max=1,
                                            step=0.001,
                                            value=0.5,
                                             marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                        ),
                                        html.Div(id='time-step-value', children="Time step: 0.5"),
                                    ],
                                    className="mb-4"
                                ),
                            ],
                            body=True,
                            className="mt-4"
                        ),
                    ],
                    width=3
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                html.H5("Locations", className="card-title"),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Name"),
                                        dbc.Input(id="input-location-name", placeholder="Enter a name", type="text"),
                                    ],
                                    className="mb-3"
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("X"),
                                        dbc.Input(id="input-x-coordinate", placeholder="Enter X-coords", type="number"),
                                    ],
                                    className="mb-3"
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Y"),
                                        dbc.Input(id="input-y-coordinate", placeholder="Enter Y-coords", type="number"),
                                    ],
                                    className="mb-3"
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.InputGroupText("Population"),
                                        dbc.Input(id="input-population", placeholder="Population", type="number"),
                                    ],
                                    className="mb-3"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.Switch(id="input-mobility-toggle", label="Mobility", value=False), width=10),
                                    ],
                                    className="mb-3"
                                ),
                                html.Div(
                                    [
                                        html.Label("Infection Source", className="mb-1"),
                                        dcc.Slider(
                                            id='input-infection-source',
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=50,
                                            marks={i: str(i) for i in range(0, 101, 25)},
                                        ),
                                        html.Div(id='infection-source-value', children="Infection source: 50"),
                                    ],
                                    className="mb-4"
                                ),
                                html.Div(
                                    [
                                        html.Label("Vaccination Rate", className="mb-1"),
                                        dcc.Slider(
                                            id='input-vaccination-rate',
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.5,
                                             marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
                                        ),
                                        html.Div(id='vaccination-rate-value', children="Vaccination rate: 0.5"),
                                    ],
                                    className="mb-4"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.Div(id='upload-data-container', children=[
                                                dcc.Upload(
                                                    id='upload-data',
                                                    children=html.Button('Upload Data', id='upload-button', className='btn btn-info', n_clicks=0),
                                                    style={'display': 'block'}, 
                                                    multiple=True
                                                )
                                            ]),
                                            width=12
                                        ),
                                    ],
                                    className="mb-4"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(html.Button('Add Location', id='add-location-btn', className='btn btn-primary', n_clicks=0), width=4),
                                        dbc.Col(html.Button('Clear Selected', id='clear-selected-btn', className='btn btn-warning', n_clicks=0), width=4),
                                        dbc.Col(html.Button('Clear All', id='clear-locations-btn', className='btn btn-danger', n_clicks=0), width=4),
                                    ],
                                    className="mb-3"
                                ),
                                html.Div(id='status-message', className="text-center"),
                            ],
                            body=True
                        ),
                    ],
                    width=3
                ),
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(label='Locations of Interest', children=[
                                    html.Div(id='locations-content', children=[
                                        dash_table.DataTable(
                                            id='locations-table',
                                            columns=[
                                                {'name': 'Name', 'id': 'name', 'editable': False},
                                                {'name': 'X-coords', 'id': 'x', 'type': 'numeric', 'editable': True},
                                                {'name': 'Y-coords', 'id': 'y', 'type': 'numeric', 'editable': True},
                                                {'name': 'Population', 'id': 'population', 'type': 'numeric', 'editable': True},
                                                {'name': 'Infection Source', 'id': 'infection_source', 'type': 'numeric', 'editable': True},
                                                {'name': 'Vaccination Rate', 'id': 'vaccination_rate', 'type': 'numeric', 'editable': True},
                                                {'name': 'Mobility', 'id': 'mobility', 'editable': True}
                                            ],
                                            editable=True,
                                            row_deletable=True,
                                            style_table={'overflowX': 'auto'},
                                            style_header={
                                                'backgroundColor': '#222831',
                                                'color': '#EEEEEE',
                                                'fontWeight': 'bold',
                                                'borderBottom': '3px solid #00ADB5',
                                                'textAlign': 'center',
                                            },
                                            style_cell={
                                                'backgroundColor': '#393E46',
                                                'color': '#EEEEEE',
                                                'border': '1px solid #222831',
                                                'textAlign': 'center',  
                                                'padding': '10px',
                                            },
                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': '#222831'
                                                }
                                            ],
                                            style_as_list_view=True,
                                            style_filter={
                                                'backgroundColor': '#393E46',
                                                'border': '1px solid #00ADB5',
                                            },
                                            style_data={
                                                'whiteSpace': 'normal',
                                                'height': 'auto',
                                            },
                                        )
                                    ])
                                ], tab_id='tab-locations'),
                                dbc.Tab(label='Nodal Analysis', children=[
                                    dcc.Loading(
                                        id="loading-stats",
                                        type="cube",
                                        children=dcc.Graph(
                                            id='statistics-plot',
                                        )
                                    )
                                ], tab_id='tab-stats'),
                               dbc.Tab(label='Map', children=[
                                    dcc.Loading(
                                        id="loading-2d-map",
                                        type="cube",
                                        children=dcc.Graph(
                                            id='map-plot', 
                                        )
                                    )
                                ], tab_id='tab-map'),

                                dbc.Tab(label='3-D Map', children=[
                                    dcc.Loading(
                                        id="loading-3d-map",
                                        type="cube",
                                        children=dcc.Graph(
                                            id='3d-map-plot', 
                                        )
                                    )
                                ], tab_id='tab-3d-map'),
                                dbc.Tab(label="Chat with Sim", children=[
                                    html.Div([
                                        dcc.Textarea(
                                            id='user-query', 
                                            placeholder='Ask a question about the simulation results...',
                                            style={
                                                'width': '100%',
                                                'height': '150px',
                                                'color': '#fff',
                                                'backgroundColor': '#2C3E50',
                                                'border': '1px solid #00ADB5',
                                                'borderRadius': '5px',
                                                'padding': '10px',  
                                                'marginBottom': '10px'  
                                            }
                                        ),
                                        html.Button(
                                            'Submit Query', 
                                            id='submit-query', 
                                            n_clicks=0,
                                            className='btn btn-primary btn-lg',
                                            style={
                                                'width': '100%',
                                                'color': '#fff',
                                                'backgroundColor': '#17a2b8',
                                                'border': 'none',
                                                'marginBottom': '10px'
                                            }
                                        ),
                                        html.Div(id='openai-response', style={
                                            'color': '#fff',  
                                            'backgroundColor': '#394E60',  
                                            'padding': '20px',  
                                            'borderRadius': '5px', 
                                            'border': '1px solid #00ADB5'  
                                        })
                                    ], style={'padding': '20px'})  
                                ], tab_id='chat-with-simulator'),
                            ],
                            id='tabs',
                            active_tab='tab-locations'
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(html.Button('Run Sim', id='run-button', n_clicks=0, className='btn btn-primary'), width=3),
                                            dbc.Col(html.Button('Abort Sim', id='abort-button', n_clicks=0, className='btn btn-danger'), width=3),
                                            dbc.Col(html.Button('Preset 1', id='preset-1', n_clicks=0, className='btn btn-dark'), width=3),
                                            dbc.Col(html.Button('Preset 2', id='preset-2', n_clicks=0, className='btn btn-dark'), width=3),
                                        ],
                                        className='mt-4'
                                    ),
                                ]
                            ),
                            className='mt-4'
                        ),
                    ],
                    md=6
                ),
            ]
        ),
    ],
    fluid=True,
    className="dbc"
)

def advanced_filter_data_based_on_query(query):
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    df_descriptions = pd.read_excel('output_with_descriptions.xlsx')
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(query.lower())
    filtered_words = [word for word in words if word not in stop_words]

    filter_mask = pd.Series([False] * len(df_descriptions))

    node_regex = '|'.join(map(re.escape, filtered_words))
    filter_mask |= df_descriptions['Node'].str.contains(node_regex, case=False, regex=True)

    conditions = ['rising', 'declining', 'improving', 'decreasing', 'stable']
    matched_conditions = [word for word in filtered_words if word in conditions]
    for condition in matched_conditions:
        if condition in ['rising', 'increasing']:
            filter_mask |= df_descriptions['Description'].str.contains('rising|increasing', case=False)
        elif condition in ['declining', 'decreasing']:
            filter_mask |= df_descriptions['Description'].str.contains('declining|decreasing', case=False)
        elif condition == 'improving':
            filter_mask |= df_descriptions['Description'].str.contains('improving', case=False)
        elif condition == 'stable':
            filter_mask |= df_descriptions['Description'].str.contains('stable', case=False)

    filtered_df = df_descriptions[filter_mask]
    if filtered_df.empty:
        return "No data matching your query was found."
    return filtered_df

def get_middle_description(s):
    n = len(s)
    middle_index = n // 2
    if n % 2 == 0:
        middle_index -= 1
    sorted_s = s.sort_values() 
    return sorted_s.iloc[middle_index]

def summarize_data_by_time_and_node(data_df, interval=10):
    data_df['Time Interval'] = (data_df['Time'] // interval) * interval
    
    summary_df = data_df.groupby(['Time Interval', 'Node']).agg({
        'Susceptible': 'mean',
        'Infected': 'mean',
        'Recovered': 'mean',
        'Description': get_middle_description  
    }).reset_index()

    summary_texts = []
    for _, row in summary_df.iterrows():
        summary = f"At time {row['Time Interval']} to {row['Time Interval'] + interval}, node {row['Node']}: {int(row['Susceptible'])} susceptible, {int(row['Infected'])} infected, and {int(row['Recovered'])} recovered. Description: {row['Description']}"
        summary_texts.append(summary)
    return " ".join(summary_texts)


def convert_data_to_text(data_df):
    if isinstance(data_df, str):  
        return data_df
    
    summary_texts = []
    for _, row in data_df.iterrows():
        summary = f"At time {row['Time']}, in node {row['Node']}, there were {row['Susceptible']} susceptible, {row['Infected']} infected, and {row['Recovered']} recovered. Description: {row['Description']}"
        summary_texts.append(summary)
    return " ".join(summary_texts)

def ask_openai(query, previous_messages):
    filtered_df = advanced_filter_data_based_on_query(query)
    if isinstance(filtered_df, str) and "No data matching" in filtered_df:
        messages = previous_messages + [{"role": "user", "content": query}]
    else:
        summarized_df = summarize_data_by_time_and_node(filtered_df)
        text_for_ai = convert_data_to_text(summarized_df)
        print("Text for AI:", text_for_ai)
        
        messages = previous_messages + [
            {"role": "system", "content": "You are an AI that assists with understanding simulation data. Answer questions based on the data. Ensure that your answer is detailed with numbers and timing based on the data. Only pick clear and important patterns. Do not provide vague or irrelevant information."},
            {"role": "user", "content": text_for_ai},
            {"role": "user", "content": query}
        ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=250,
        temperature=0.7
    )
    
    if response.choices:
        ai_response = response.choices[0].message.content.strip()
    else:
        ai_response = "No response generated."

    previous_messages.append({"role": "user", "content": query})
    previous_messages.append({"role": "assistant", "content": ai_response})

    return ai_response, previous_messages

@app.callback(
    Output('openai-response', 'children'),
    Output('conversation-history', 'data'),
    Input('submit-query', 'n_clicks'),
    State('user-query', 'value'),
    State('conversation-history', 'data'),
    State('tabs', 'active_tab')
)
def handle_query(n_clicks, query, history, active_tab):
    if active_tab != 'chat-with-simulator' or n_clicks < 1 or not query:
        raise dash.exceptions.PreventUpdate
    
    response, updated_history = ask_openai(query, history)
    return response, updated_history


if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
    print("Server stopped.")
