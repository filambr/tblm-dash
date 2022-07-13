from call_backs import *
import json
import pandas as pd
import plotly.express as px
from jupyter_dash import JupyterDash
from dash import Dash
# import dash_core_components as dcc
# import dash_html_components as html
from dash import html, dash_table, dcc
# import dash_table
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.long_callback import DiskcacheLongCallbackManager
import base64
import datetime
import io
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import diskcache
from dash.dash_table.Format import Format, Scheme, Group
from zipfile import ZipFile
import dash_daq as daq

from src import *
from src.tblm_model import Model

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.DataFrame({'Cm': [1*10**-6], 'CH': [8*10**-6], 'r0': [5*10**-7], 'rho': [10**4.5], 'Ydef': [50], 'Rsol': [0.1], 'd_sub': [1.8*10**-7]})
df_status = pd.DataFrame({'Cm': ['False'], 'CH': ['True'], 'r0': ['True'], 'rho': ['False'], 'Ydef': ['False'], 'Rsol': ['True'], 'd_sub':['False']})
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True, long_callback_manager=long_callback_manager)

colors = px.colors.qualitative.Plotly


app.layout = html.Div([
    dcc.Store(id='Data', data=None),
    dcc.Store(id='btn-par-val-clicks'),
    dcc.Store(id='btn-par-stat-clicks'),
    dcc.Store(id='saved-f-range'),
    dcc.Store(id='saved-N-range'),
    dcc.Store(id='fit-Data', data=None),
    dcc.Store(id='fit-pdf',data=None),
    dcc.Store(id ='fited-kwargs', data=None),
###########################################################
    html.Div([
            html.Div([dash_table.DataTable(data=df.to_dict('records'),
                                         columns=[{"name": i, "id": i, 'type':'text'} for i in df.columns],
                                         id='tbl-par-val', editable=True)
                     ], style={'display': 'inline-block', 'width':'80%'},
            ),
            html.Div([html.Button('Add Row', id='btn-par-val', n_clicks=0),
                     ], style={'display': 'inline-block', 'width':'15%','vertical-align': 'bottom', 'margin':'10px'},
            ),

    ], style={'width':'90%', 'vertical-align': 'bottom'}),
###########################################################
    html.Div([
            html.Div([dash_table.DataTable(data=df_status.to_dict('records'),
                                         columns=[{"name": i, "id": i, 'type':'text'} for i in df.columns],
                                         id='tbl-par-stat', editable=True)
                     ], style={'display': 'inline-block', 'width':'80%'},
            ),
            html.Div([html.Button('Add Row', id='btn-par-stat', n_clicks=0),
                     ], style={'display': 'inline-block', 'width':'15%','vertical-align': 'bottom', 'margin':'10px'},
            ),

    ], style={'width':'90%', 'vertical-align': 'bottom'}),

    html.Div([
########################################################
        html.Div([
            html.Div([
                html.Div([html.Button("Download", id="btn_download", n_clicks=0),
                          dcc.Download(id="download_zip"),
                         ], style={'display': 'inline-block', 'margin-left':'20px', 'margin-top':'20px'},
                ),
                html.Div([html.Button('Display Data', id='display-data', n_clicks=0),
                         ], style={'display': 'inline-block', 'margin-left':'20px', 'margin-top':'20px'},
                ),
                html.Div([html.Button('Display Fit', id='display-fit', n_clicks=0),
                         ], style={'display': 'inline-block', 'margin-left':'20px', 'margin-top':'20px'},
                ),
                html.Div([html.Button('Fit', id='fit', n_clicks=0),
                         ], style={'display': 'inline-block', 'margin-left':'20px', 'margin-top':'20px'},
                ),
                html.Div([daq.Indicator(label='indicator', id='fit-indicator', value=False),
                         ], style={'display': 'inline-block', 'margin-left':'20px', 'margin-top':'20px'},
                ),
                html.Div([daq.NumericInput(label='Enter Area in cm', id='area', value=1),
                         ], style={'display': 'inline-block', 'margin-left':'20px', 'margin-top':'20px'},
                ),

                html.Div(children=[
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files'),
                            ]),
                        style={
                            'width': '300px',
                            'height': '35px',
                            'lineHeight': '35px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            "border-radius": "5px",
                            'textAlign': 'center',
                            'display': 'inline-block',
                            "font-family": "Arial, Helvetica, sans-serif",
                            'font-weight': 'normal',
                            'font-size': '16px',
                            'margin-left':'20px',
                            'margin-top':'20px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=True
                    ),
                ],
                style={
                      "display": "inline-block",
                  },
        ),
                html.Div([dcc.Dropdown(id='algorithm',
   options=[
       {'label': 'Powell', 'value': 'Powell'},
       {'label': 'Differential evolution', 'value': 'Differential evolution'},
   ],
   value='Powell'
),
                         ], style={'display': 'inline-block', 'margin-left':'20px', 'margin-top':'20px', 'width':'200px', 'heigth':'35'},
                ),

            ]),
            html.Div([
            html.Div([dcc.Graph(id='g1')
                ], style={'display': 'inline-block', 'width':'50%'}),
            html.Div([dcc.Graph(id='g2')
                ], style={'display': 'inline-block', 'width':'50%'}),
            ]),
            html.Div([
            html.Div([dcc.Graph(id='g3')
                ], style={'display': 'inline-block', 'width':'50%'}),
            html.Div([dcc.Graph(id='g4')
                ], style={'display': 'inline-block', 'width':'50%'}),
            ]),
        ], style={'display':'inline-block', 'width':'60%'}
    ),
    ###########################################################
        html.Div([
            html.H4('Constraint sliders'),
            html.Div([html.H6('log(f) range'),
                dcc.RangeSlider(min=-2.01, max=6.01, step=0.01,
                id='slider-f',
                marks={i: "{:10.0f}".format(i) for i in [-2,0,2,4,6]},
                value=(-2, 6),
                updatemode='drag'
                )], style={'margin-top':'10px'}),
            html.Div([html.H6('log(N) range'),
                dcc.RangeSlider(min=-3, max=3, step=0.01,
                id='slider-N',
                marks={i: "{:10.0f}".format(i) for i in [-3,-2,-1,0,1,2,3]},
                value=(-2, 2.5),
                updatemode='drag'
                )], style={'margin-top':'10px'}),
            html.Div([html.H6('log(alpha)'),
                dcc.Slider(min=-3, max=3, step=0.01,
                id='slider-alpha',
                marks={i: "{:10.0f}".format(i) for i in [-3,-2,-1,0,1,2,3]},
                value=-1,
                updatemode='drag'
                )], style={'margin-top':'10px'}),
            html.H4('Parameter bound sliders'),
            html.Div([html.H6('Cm'),
                dcc.RangeSlider(min=0.3, max=1.5, step=0.01,
                id='slider-Cm',
                marks={i: "{:10.2f}".format(i) for i in np.linspace(0.3,1.5,6)},
                value=(0.5, 1),
                updatemode='drag'
                )], style={'margin-top':'10px'}),
            html.Div([html.H6('CH'),
                dcc.RangeSlider(min=3, max=25, step=0.01,
                id='slider-CH',
                marks={i: "{:10.2f}".format(i) for i in np.linspace(3,25,11)},
                value=(5, 20),
                updatemode='drag'
                )], style={'margin-top':'10px'}),
            html.Div([html.H6('r0'),
                dcc.RangeSlider(min=0.5, max=30, step=0.01,
                id='slider-r0',
                marks={i: "{:10.2f}".format(i) for i in np.linspace(0.5,30,6)},
                value=(1, 20),
                updatemode='drag'
                )], style={'margin-top':'10px'}),
            html.Div([html.H6('Ydef'),
                dcc.RangeSlider(min=3, max=500, step=0.01,
                id='slider-Ydef',
                marks={i: "{:10.2f}".format(i) for i in np.linspace(3,500,6)},
                value=(10, 200),
                updatemode='drag'
                )], style={'margin-top':'10px'}),
            html.Div([html.H6('rho'),
                dcc.RangeSlider(min=0, max=8, step=10,
                id='slider-rho',
                marks={i: "{:10.0f}".format(10 ** i) for i in np.linspace(0,8,6)},
                value=(3, 6),
                updatemode='drag'
                )], style={'margin-top':'10px'}),
            html.Div([html.H6('Rsol'),
                dcc.RangeSlider(min=0, max=200, step=0.01,
                id='slider-Rsol',
                marks={i: "{:10.2f}".format(i) for i in [0, 50, 100, 150, 200]},
                value=(10, 100),
                updatemode='drag',
                )], style={'margin-top':'10px'}),
            html.Div([html.H6('d_sub'),
                dcc.RangeSlider(min=0, max=3, step=0.01,
                id='slider-d_sub',
                marks={i: "{:10.2f}".format(i) for i in np.linspace(1,3,6)},
                value=(1, 2),
                updatemode='drag',
                )], style={'margin-top':'10px'}),
        ], style={'display':'inline-block','width':'25%','vertical-align':'top'}
    ),
###########################################################
])
])
@app.callback(
    Output('area', 'value'),
    Input('area', 'value')
)
def update_output(value):
    print(value)
    return value

@app.callback(Output('fit-indicator','value'),Input('fit-Data', 'data'))
def indicator(data):
    if data is not None:
        return True
    else:
        return False

@app.callback(
    Output("download_zip", "data"),
    Input("btn_download", "n_clicks"),
    State('fit-Data', 'data'),
    State('fit-pdf', 'data'),
    State('fited-kwargs', 'data'),
    prevent_initial_call=True,
)
def func(n_clicks,spectra, pdfs, kwargs):
    if n_clicks>0:
        print(pdfs)
        with ZipFile('sample2.zip', 'w') as zipObj2:
            for i, spectrum in enumerate(spectra):
                df = pd.DataFrame(spectrum)
                df.to_csv(f'spectrum_{i}.csv')
                zipObj2.write(f'spectrum_{i}.csv')

            df_pdf = pd.DataFrame(pdfs[0])
            df_pdf.to_csv('pdf.csv')
            zipObj2.write('pdf.csv')

            df_kwargs = pd.DataFrame(kwargs)
            df_kwargs.to_csv('params.csv')
            zipObj2.write('params.csv')
            zipObj2.close()
        return dcc.send_file('sample2.zip')
    else:
        raise PreventUpdate

add_row(app)
# update_data(app)
plot_data(app)
plot_fit(app)
fit(app)
app.run_server(debug=True)
