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
from sim_class import Simulator
import supporting_functions as sf
import plots_functions as pf

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

@app.callback(
    Output('location-data-store', 'data'),
    [
        Input('upload-data', 'contents'),
        Input('add-location-btn', 'n_clicks'),
        Input('clear-locations-btn', 'n_clicks'),
        Input('locations-table', 'data_timestamp')
    ],
    [
        State('upload-data', 'filename'),
        State('input-location-name', 'value'),
        State('input-x-coordinate', 'value'),
        State('input-y-coordinate', 'value'),
        State('input-population', 'value'),
        State('input-infection-source', 'value'),
        State('input-vaccination-rate', 'value'),
        State('input-mobility-toggle', 'value'),
        State('location-data-store', 'data'),
        State('locations-table', 'data')
    ]
)
def unified_data_handler(contents, add_clicks, clear_clicks, table_timestamp,
                         filenames, name, x, y, population, infection_source,
                         vaccination_rate, mobility, existing_data, table_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'upload-data' and contents:
        combined_data = existing_data[:]
        for content, filename in zip(contents, filenames):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            if 'xlsx' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
                new_data = sf.standardize_data_keys(df.to_dict('records'))
                combined_data += new_data
        return combined_data

    elif triggered_id == 'add-location-btn' and name:
        mobility_text = "Yes" if mobility else "No"
        new_entry = sf.standardize_data_keys([{
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

    if triggered_id == 'locations-table':
        return table_data

    return existing_data


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

    sim_params = pf.setup_simulation_parameters(alpha, beta, kappa, location_data)
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

    df_descriptions = sf.add_detailed_descriptions(df)

    df_descriptions.to_excel('output_with_descriptions.xlsx', index=False)

    map_fig = pf.setup_map_figure(sim_params, timesteps, state_trajectories, tau_values)
    map_fig3d = pf.setup_3dmap_figure(sim_params, timesteps, state_trajectories, tau_values)
    stats_fig = pf.setup_stats_figure(sim_params, state_trajectories, timesteps)

    return map_fig, map_fig3d, stats_fig

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
        dcc.Store(id='processing-state', data={'is_processing': False}), 
        html.Div(id='spinner', className='invisible-spinner'), 
        html.Div(id='button-clicked', style={'display': 'none'}), 
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
                                            "Submit Query",
                                            id='submit-query',
                                            n_clicks=0,
                                            className='btn btn-primary btn-lg',
                                            style={'width': '100%', 'textAlign': 'left', 'marginBottom': '10px'}
                                        ),
                                        dcc.Loading(
                                            id="loading-openai-response",
                                            type="circle",
                                            children=[
                                                html.Div(id='openai-response', style={
                                                    'color': '#fff',
                                                    'backgroundColor': '#394E60',
                                                    'padding': '20px',
                                                    'borderRadius': '5px',
                                                    'border': '1px solid #00ADB5',
                                                    'marginTop': '10px'
                                                })
                                            ]
                                        )
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
    
    response, updated_history = sf.ask_openai(query, history)
    return response, updated_history

if __name__ == '__main__':
    app.run_server(debug=True, host='localhost', port=8050)

