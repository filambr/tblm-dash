import json
import pandas as pd
import plotly.express as px
from jupyter_dash import JupyterDash
from dash import Dash
# import dash_core_components as dcc
# import dash_html_components as html
from dash import html, dash_table, dcc, callback_context
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
colors = px.colors.qualitative.Plotly


def parse_contents(contents, filename, date, area):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename or 'txt' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),
                            delimiter = '\t', names=['f', 'zr', 'zi'])

            df0 = pd.DataFrame({'f':df['f'], 'zr':df['zr']*area, 'zi':df['zi']*area,
                                })

            dff = eis_df(df0)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return dff.to_dict('list')

def add_row(app):
    @app.callback([Output('tbl-par-val', 'data'), Output('btn-par-val-clicks','data')], [State('tbl-par-val', 'columns'), State('tbl-par-val', 'data'),
                                          Input('btn-par-val','n_clicks'), State('btn-par-val-clicks','data')])
    def add_row_init_values(columns, rows, n_clicks, click_count):
        updated_click_count = 0
        if click_count is None:
            if n_clicks > 0:
    #             rows.append({c['id']: 0 for c in columns})
                rows.append(rows[-1])
                updated_click_count = n_clicks
            else:
                updated_click_count = n_clicks
        elif n_clicks > click_count:
    #         rows.append({c['id']: 0 for c in columns})
            rows.append(rows[-1])
            updated_click_count = n_clicks
        return rows, updated_click_count


    @app.callback([Output('tbl-par-stat', 'data'), Output('btn-par-stat-clicks','data')],
                  [State('tbl-par-stat', 'columns'),State('tbl-par-stat', 'data'),
                  Input('btn-par-stat','n_clicks'), State('btn-par-stat-clicks','data')])
    def add_row_status(columns, rows, n_clicks, click_count):
        updated_click_count = 0
        if click_count is None:
            if n_clicks > 0:
                rows.append(rows[-1])
                updated_click_count = n_clicks
            else:
                updated_click_count = n_clicks
        elif n_clicks > click_count:
            rows.append(rows[-1])
            updated_click_count = n_clicks
        return rows, updated_click_count


def fig_nice(fig):
    font_dict=dict(family='Arial',
                   size=14,
                   color='black'
                  )
    fig.update_layout(font=font_dict,
                     plot_bgcolor='white',
                     width=85*5,
                     height=70*5,
                     margin=dict(l=10, r=50, t=20, b=10),
                    #  showlegend=False
                     )

    fig.update_yaxes(showline=True,
                     linecolor='black',
                     linewidth=1.2,
                     ticks='inside',
                     mirror='allticks',
                     tickwidth=1.2,
                     tickcolor='black',
                     )
    fig.update_xaxes(showline=True,
                     linecolor='black',
                     linewidth=1.2,
                     ticks='inside',
                     mirror='allticks',
                     tickwidth=1.2,
                     tickcolor='black',
                     )
    return fig


def fit_tblm_data(kwargs, kwargs_status, f_range, n_range, data, alp, algorithm, bmin, bmax):
#     if n_clicks >0:

    try:
        f_range = {'min':min(f_range),'max':max(f_range)}
        n_range = {'min':min(n_range),'max':max(n_range)}
        number_N = len(data[0]['f'])+5
        logNs = N_range(number_N, logNmin=n_range['min'], logNmax=n_range['max'])
        f = []
        Y = []
        for i, dat in enumerate(data):
            df = pd.DataFrame(dat)
            f_temp = df['f'][(df['f']>10**f_range['min']) & (df['f']<10**f_range['max'])].to_numpy()
            Zre = df['zreal'][(df['f']>10**f_range['min']) & (df['f']<10**f_range['max'])].to_numpy()
            Zim = df['zimag'][(df['f']>10**f_range['min']) & (df['f']<10**f_range['max'])].to_numpy()
            Y_temp = 1/(Zre + 1j*Zim)
            f.append(f_temp)
            Y.append(Y_temp)

        logN = [logNs for i in data]
        alpha = [10**alp for i in data]
        gener_dist = [None for i in data]
        reg_matx = [regularization_matx(number_N) for i in data]
        df_args = pd.DataFrame({'f': f,
                                'Y': Y,
                                'gener_dist': gener_dist,
                                'logN': logN,
                                'alpha': alpha,
                                'reg_matx': reg_matx


        })

        df_kwargs = pd.DataFrame({x: kwargs_i[x] for x in kwargs_i} for kwargs_i in kwargs)
        df_kwargs_status = pd.DataFrame({x: str(kwargs_status_i[x])
                                         for x in kwargs_status_i
                                        }
                                        for kwargs_status_i in kwargs_status
                           )

        model = Model()
        fited_data, pdf, fitted_kwargs = model.fit(kwargs=df_kwargs,
                                         kwargs_status=df_kwargs_status,
                                         args=df_args, b_min=bmin, b_max=bmax, 
                                         algorithm=algorithm)

        return [d.to_dict() for d in fited_data], pdf, fitted_kwargs#,[model.pdf.to_dict()]
    except Exception as e:
        print(str(e), 'fit-data-funciton-exception')
        return html.Div(str(e))
#     else:
#         raise PreventUpdate


def append_phase(fig, df, row, col, color):
    fig.append_trace({'x':df['f'],'y':-df['zphase'],'type':'scatter', 'line':{'color':color}},row=row,col=col)
    fig.update_xaxes(title_text="f, Hz",type="log", row=row, col=col)
    fig.update_yaxes(title_text=r"$\text{argY}_{\text{phase}}, \text{deg}^o$", row=row, col=col)
    return fig

def append_abs(fig, df, row, col, color):
    fig.append_trace({'x':df['f'],'y':1/df['zabs'],'type':'scatter', 'line':{'color':color}},row=row,col=col)
    fig.update_xaxes(title_text="f, Hz", type="log", row=row, col=col)
    fig.update_yaxes(title_text="|Y|, S", type="log", row=row, col=col)
    return fig

def append_cole(fig, df, row, col, color):
    fig.append_trace({'x':df['creal']*10**6,'y':-df['cimag']*10**6,'type':'scatter', 'line':{'color':color}},row=row,col=col)
    fig.update_xaxes(title_text=r'$\text{C}_{\text{real }} \mu F$', row=row, col=col)
    fig.update_yaxes(title_text=r'$\text{-C}_{\text{imag }} \mu F$', row=row, col=col)
    return fig



def append_phase_scatter(fig, df, color):
    fig.add_trace({'x':df['f'],'y':-df['zphase'],
                      'type':'scatter', 'mode':'markers',
                      'marker':{'color':'white','line_width':1.5,
                                'line':{'color':color}
                               }
                     },
                    )

    fig.update_xaxes(title_text="f, Hz",type="log")
    fig.update_yaxes(title_text=r"argYphase, deg")
    return fig

def append_abs_scatter(fig, df, color):
    fig.add_trace({'x':df['f'],'y':1/df['zabs'],
                      'type':'scatter', 'mode':'markers',
                      'marker':{'color':'white','line_width':1.5,
                                'line':{'color':color}
                               }
                     },
                    )
    fig.update_xaxes(title_text="f, Hz", type="log")
    fig.update_yaxes(title_text="|Y|, S", type="log")
    return fig

def append_cole_scatter(fig, df, color):
    fig.add_trace({'x':df['creal']*10**6,'y':-df['cimag']*10**6,
                  'type':'scatter', 'mode':'markers',
                  'marker':{'color':'white','line_width':1.5,
                            'line':{'color':color}
                           }
                 },
                )
    fig.update_xaxes(title_text=r'Creal, uF/cm2',
                    )
    fig.update_yaxes(title_text=r'-Cimag, uF/cm2',
                     scaleanchor = "x", scaleratio = 1,
                    )
    return fig

def append_phase_plot(fig, df, color):
    fig.add_trace({'x':df['f'],'y':-df['zphase'],
                      'type':'scatter', 'mode':'lines',
                   'marker':{'color':color,'line_width':1.5,
                            'line':{'color':color}
                           }
                     },
                    )

    fig.update_xaxes(title_text="f, Hz",type="log")
    fig.update_yaxes(title_text=r"argYphase, deg")
    return fig

def append_abs_plot(fig, df, color):
    fig.add_trace({'x':df['f'],'y':1/df['zabs'],
                      'type':'scatter', 'mode':'lines',
                   'marker':{'color':color,'line_width':1.5,
                            'line':{'color':color}
                           }

                     },
                    )
    fig.update_xaxes(title_text="f, Hz", type="log")
    fig.update_yaxes(title_text="|Y|, S", type="log")
    return fig

def append_cole_plot(fig, df, color):
    fig.add_trace({'x':df['creal']*10**6,'y':-df['cimag']*10**6,
                  'type':'scatter', 'mode':'lines',
                   'marker':{'color':color,'line_width':1.5,
                            'line':{'color':color}
                           }
                 },
                )
    fig.update_xaxes(title_text=r'Creal, uF/cm2',
                    )
    fig.update_yaxes(title_text=r'-Cimag, uF/cm2',
                     scaleanchor = "x", scaleratio = 1,
                    )
    return fig

def append_pdf_plot(fig, df, color):
    fig.add_trace({'x':df['logN'],'y':df['dist']/(df['logN'][1]-df['logN'][0]),
                  'type':'scatter', 'mode':'lines',
                   'marker':{'color':color,'line_width':1.5,
                            'line':{'color':color}
                           }
                 },
                )
    fig.update_xaxes(title_text=r'log10(Ndef)',
                    )
    fig.update_yaxes(title_text=r'Probability density',
                     scaleanchor = "x", scaleratio = 1,
                    )
    return fig

def update_contents(list_of_contents, list_of_names, list_of_dates, area):
    if list_of_contents is not None:
        children = [parse_contents(c, n, d, area) for c, n, d in
        zip(list_of_contents, list_of_names, list_of_dates)]
        return children

    
    
    
def master_callback(app):
    @app.callback(Output('g1', 'figure'),
                  Output('g2', 'figure'),
                  Output('g3', 'figure'),
                  Output('g4', 'figure'),
                  Output('display-data', 'children'),
                  Output('Data', 'data'),
                  Output('fit-Data','data'),
                  Output('fit-pdf','data'),

#                   Output('clear', 'value'),
                  Input('display-data','n_clicks'),
                  Input('slider-f', 'value'),
                  Input('clear', 'n_clicks'),
                  Input('fit', 'n_clicks'),
                  
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('upload-data', 'last_modified'),
                  State('area', 'value'),
                  
                  State('fit-pdf','data'),                  
                  State('fit-Data','data'),
                  State('Data','data'),
                  State('g1', 'figure'),
                  State('g2', 'figure'),
                  State('g3', 'figure'),
                  State('g4', 'figure'),
                  
                 State('tbl-par-val','data'),
                 State('tbl-par-stat','data'),
#                  State('slider-f', 'value'),
                 State('slider-N', 'value'),
#                  State('Data', 'data'),
                 State('slider-alpha','value'),
                 State('algorithm','value'), 
                  
                 State('bmin', 'data'),
                 State('bmax', 'data')
                 )
    def master(n, value, btn_clear, btn_fit,
                     list_of_contents, list_of_names, list_of_dates, area,
                     fit_pdf, fit_data, data,
                     figure1, figure2, figure3, figure4,
                     kwargs, kwargs_status, n_range, alp, algorithm,
                     bmin, bmax
                    ):
        if bmin is not None and bmax is not None:
            b_min = pd.DataFrame({'Cm':bmin['Cm'],
                                  'CH':bmin['CH'],
                                  'r0':bmin['r0'],
                                  'rho':bmin['rho'],
                                  'Ydef':bmin['Ydef'],
                                  'Rsol':bmin['Rsol'],
                                  'd_sub':bmin['d_sub']})

            b_max = pd.DataFrame({'Cm':bmax['Cm'],
                                  'CH':bmax['CH'],
                                  'r0':bmax['r0'],
                                  'rho':bmax['rho'],
                                  'Ydef':bmax['Ydef'],
                                  'Rsol':bmax['Rsol'],
                                  'd_sub':bmax['d_sub']})
            
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if not changed_id:
            raise PreventUpdate  
        
        if 'fit' in changed_id:
            fit_data, fit_pdf, fitted_kwargs = fit_tblm_data(kwargs, kwargs_status, value, n_range, data, alp, algorithm, b_min, b_max)
#             n_clicks, kwargs, kwargs_status, f_range, n_range, data, alp, algorithm
        if 'clear' in changed_id:
            print('removing uploaded data')
            figure1 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure2 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure3 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure4 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure1 = fig_nice(figure1)
            figure2 = fig_nice(figure2)
            figure3 = fig_nice(figure3)
            figure4 = fig_nice(figure4)
            return figure1, figure2, figure3, figure4, 'Plot Data', None, None, None
        if 'upload-data' in changed_id:
            print('trying to upload data')

            children = update_contents(list_of_contents, list_of_names, list_of_dates, area)
#             children = update_output(list_of_contents, list_of_names, list_of_dates, area)
            print('succesfuly parsed contents')
            figure1 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure2 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure3 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure4 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure1 = fig_nice(figure1)
            figure2 = fig_nice(figure2)
            figure3 = fig_nice(figure3)
            figure4 = fig_nice(figure4)
            return figure1, figure2, figure3, figure4, 'Plot Data', children, None, None
        
        elif figure1 is None:
            figure1 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure2 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure3 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)
            figure4 = make_subplots(rows=1, cols=1,
                                   horizontal_spacing=0.15,
                                   vertical_spacing=0.15)        
        if data is not None:

            figure1=go.Figure()
            figure2=go.Figure()
            figure3=go.Figure()
            figure4=go.Figure()

            for i, dat in enumerate(data):
                df = pd.DataFrame(dat)

                f_range = {'min':min(value),'max':max(value)}
                df = df[(df['f']>10**f_range['min']) & (df['f']<10**f_range['max'])]
                figure1 = append_phase_scatter(figure1, df, colors[i])
                figure2 = append_abs_scatter(figure2, df, colors[i])
                figure3 = append_cole_scatter(figure3, df, colors[i])
            if fit_data is not None:
                for j, dat in enumerate(fit_data):
                    df = pd.DataFrame(dat)
                    df_pdf = pd.DataFrame(fit_pdf[0])
                    f_range = {'min':min(value),'max':max(value)}
                    df = df[(df['f']>10**f_range['min']) & (df['f']<10**f_range['max'])]
                    figure1 = append_phase_plot(figure1, df, colors[j])
                    figure2 = append_abs_plot(figure2, df, colors[j])
                    figure3 = append_cole_plot(figure3, df, colors[j])
                    figure4 = append_pdf_plot(figure4, df_pdf, colors[j])

            figure1.update_layout(showlegend=False)
            figure2.update_layout(showlegend=False)
            figure3.update_layout(showlegend=False)
            figure4.update_layout(showlegend=False)

            figure1 = fig_nice(figure1)
            figure2 = fig_nice(figure2)
            figure3 = fig_nice(figure3)
            figure4 = fig_nice(figure4)
            

            return figure1, figure2, figure3, figure4, 'Replot Data', data, fit_data, fit_pdf
        else:
            raise PreventUpdate


# def plot_fit(app):
#     @app.callback(Output('g4', 'figure'),
#               Input('display-fit','n_clicks'),
#               State('fit-pdf','data'),
#               State('g4', 'figure'),
#              )
#     def plot_data_g4(n, fit_pdf, figure4):
#         if n is None:
#             print(type(n))
#             raise PreventUpdate
#         elif figure4 is None:
#             figure4 = make_subplots(rows=1, cols=1,
#                                    horizontal_spacing=0.15,
#                                    vertical_spacing=0.15)

#         if fit_pdf is not None:
#             figure4=go.Figure()
#             for i, dat in enumerate(fit_pdf):
#                 df = pd.DataFrame(dat)
#                 figure4 = append_pdf_plot(figure4, df, colors[i])

#             figure4.update_layout(showlegend=False)
#             figure4 = fig_nice(figure4)
#             return figure4
#         else:
#             raise PreventUpdate

# def plot_data(app):
#     @app.callback(Output('g1', 'figure'),
#                   Output('g2', 'figure'),
#                   Output('g3', 'figure'),
#                   Output('display-data', 'children'),
#                   Input('display-data','n_clicks'),
#                   Input('slider-f', 'value'),
#                   State('fit-Data','data'),
#                   State('Data','data'),
#                   State('g1', 'figure'),
#                   State('g2', 'figure'),
#                   State('g3', 'figure'),
#                  )
#     def plot_data_g1(n, value, fit_data, data, figure1, figure2, figure3):
#         if n is None:
#             print(type(n))
#             raise PreventUpdate
#         elif figure1 is None:
#             figure1 = make_subplots(rows=1, cols=1,
#                                    horizontal_spacing=0.15,
#                                    vertical_spacing=0.15)
#             figure2 = make_subplots(rows=1, cols=1,
#                                    horizontal_spacing=0.15,
#                                    vertical_spacing=0.15)
#             figure3 = make_subplots(rows=1, cols=1,
#                                    horizontal_spacing=0.15,
#                                    vertical_spacing=0.15)

#         if data is not None:

#             figure1=go.Figure()
#             figure2=go.Figure()
#             figure3=go.Figure()
#             for i, dat in enumerate(data):
#                 df = pd.DataFrame(dat)

#                 f_range = {'min':min(value),'max':max(value)}
#                 df = df[(df['f']>10**f_range['min']) & (df['f']<10**f_range['max'])]
#                 figure1 = append_phase_scatter(figure1, df, colors[i])
#                 figure2 = append_abs_scatter(figure2, df, colors[i])
#                 figure3 = append_cole_scatter(figure3, df, colors[i])
#             if fit_data is not None:
#                 for j, dat in enumerate(fit_data):
#                     df = pd.DataFrame(dat)

#                     f_range = {'min':min(value),'max':max(value)}
#                     df = df[(df['f']>10**f_range['min']) & (df['f']<10**f_range['max'])]
#                     figure1 = append_phase_plot(figure1, df, colors[j])
#                     figure2 = append_abs_plot(figure2, df, colors[j])
#                     figure3 = append_cole_plot(figure3, df, colors[j])

#             figure1.update_layout(showlegend=False)
#             figure2.update_layout(showlegend=False)
#             figure3.update_layout(showlegend=False)
#             figure1 = fig_nice(figure1)
#             figure2 = fig_nice(figure2)
#             figure3 = fig_nice(figure3)
#             return figure1, figure2, figure3, 'Replot Data'
#         else:
#             raise PreventUpdate

# callbacks
# def update_data(app):
#     @app.callback(Output('Data', 'data'),
#                   Input('upload-data', 'contents'),
#                   State('upload-data', 'filename'),
#                   State('upload-data', 'last_modified'),
#                   State('area', 'value')
#                   )
#     def update_output(list_of_contents, list_of_names, list_of_dates, area):
#         if list_of_contents is not None:
#             children = [
#                 parse_contents(c, n, d, area) for c, n, d in
#                 zip(list_of_contents, list_of_names, list_of_dates)]
#             return children



# def fit(app):

#     @app.callback(Output('fit-Data', 'data'), Output('fit-pdf', 'data'), Output('fited-kwargs', 'data'),
#                         [Input('fit', 'n_clicks'),
#                         State('tbl-par-val','data'),
#                         State('tbl-par-stat','data'),
#                         State('slider-f', 'value'),
#                         State('slider-N', 'value'),
#                         State('Data', 'data'),
#                         State('slider-alpha','value'),
#                         State('algorithm','value')
#                         ],
#                         # running=[(Output("fit-btn", "disabled"), True, False),
#                         # ],
#     #                     manager=long_callback_manager,
#                      )

