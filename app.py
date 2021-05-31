import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import lightgbm as lgb
import pickle
import shap
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets=["https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/cyborg/bootstrap.min.css"]
external_stylesheets=["https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/united/bootstrap.min.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
#%%# Load data and models
    
with open('models/lgb_classifier.pkl', 'rb') as f:
    gbm = pickle.load(f)   
with open('models/shap_explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)   
with open('data/plotly_df.pkl', 'rb') as f:
    plotly_df = pickle.load(f) 
with open('data/original_df.pkl', 'rb') as f:
    original_df = pickle.load(f)


#%%

fig = px.scatter_mapbox(original_df.reset_index(),
                        custom_data=["id"],
                        lat="latitude", lon="longitude", 
                        color="status_group",
                        color_continuous_scale=px.colors.cyclical.IceFire, 
                        size_max=15, zoom=5,
                        mapbox_style="open-street-map",
                        hover_data  = ["construction_year"])

# fig.update_layout(height=700, width=1000) 
fig.update_layout() 
# barfig = px.bar(None, x=[43, 43], y=[34,53], color="medal", title="Long-Form Input")
 
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}         

#%%
app.layout = html.Div([
        html.H2('Tanzanian Waterpumps',
                style={
                'textAlign': 'center',
                'color': colors['text']
            }),
        html.Div(children='A dashboard for Predictive Maintenance.', style={
            'textAlign': 'center',
            'color': colors['text']
        }),

                html.Div(children=[     
                    dcc.Graph(
                    id='map',
                    figure=fig
                    ),

                    html.Div(id='display-value'),
                    html.H4('Properties of the Waterpump'),
                    dash_table.DataTable(
                    id='table',
                    data=None,
                    style_table={'overflowX': 'auto'},
                    )

                    # dcc.Graph(
                    # id='bar')
                    # figure=barfig
                    ], style={'display': 'inline-block', "width": "90%", "align-items": "center", "justify-content": "center", 'margin-bottom': 50}),
        html.H2(),
        html.H4('ML model predictions'),
        html.Div(id='predictions', style={'display': 'inline-block', "width": "90%", "align-items": "center", "justify-content": "center", 
                                                'textAlign': 'center', 'margin-bottom': 50,
                                                'color': colors['background']}, children="Select a waterpump on the map"),
        html.H2(),   
        html.H4('SHAP values for prediction explanation'),
        html.Div(children=[  
        dcc.Dropdown(
            id='dropdown',
            options=[{'label': x, 'value': i} for i, x in enumerate(['Prediction: Functional', 'Prediction: Functional Needs Repair', 'Prediction: Non-functional'])],
            # value=None
        )], style={'display': 'inline-block', "width": "30%", "align-items": "center", "justify-content": "center"}),
     
    html.Div(id='output_shap', style={'display': 'inline-block', "width": "90%", "align-items": "center", "justify-content": "center"}),

    html.Div([], id='value-container', style={'display': 'none'})
])

# @app.callback(dash.dependencies.Output('display-value', 'children'),
#               [dash.dependencies.Input('dropdown', 'value')])
# def display_value(value):
#     return 'You have selected "{}"'.format(value)

@app.callback(
    Output('dropdown', 'value'),
    Output('value-container', 'children'),
    Output('table', 'data'),
    Output('table', 'columns'),
    Output('predictions', 'children'),
    Input('map', 'clickData'))
def make_prediction(click):
    if click is None:
        raise PreventUpdate
    else:
        
        selected = click['points'][0]['customdata'][0]
        sample = plotly_df.loc[selected][:-1].values.reshape(1, -1)
        
        original = original_df.loc[selected]
        
        prediction = gbm.predict(sample)
        label = prediction.argmax(1).astype(int)[0]
        print(label)

        data = pd.DataFrame(original).T.to_dict('records')
        columns=[{"name": i, "id": i} for i in original_df.columns]

        return label, selected, data, columns, f"Predicted probabilities: \t Functional: {round(prediction[0][0]*100,1)}%, \t Needs repair: {round(prediction[0][1]*100,1)}%, \t Non-functional: {round(prediction[0][2]*100,1)}%"

        # return force_plot_html(explainer.expected_value[label], shap_values[label]), 'You have selected "{}"'.format(original)
        
        [force_plot_html(explainer.expected_value[i], shap_values[i]) for i in range(3)]
        # return 'You have selected "{}"'.format(prediction), 'You have selected "{}"'.format(prediction)

@app.callback(
    Output('output_shap', 'children'),
    Input('dropdown', 'value'),
    Input('value-container', 'children'))
def make_shap(label, selected):
    if selected is None:
        raise PreventUpdate
    elif label is None:
        raise PreventUpdate
    else:
        
        print(selected)
        sample = plotly_df.loc[selected][:-1].values.reshape(1, -1)
        shap_values = explainer.shap_values(sample)

        return force_plot_html(explainer.expected_value[label], shap_values[label])


def force_plot_html(*args):
    force_plot = shap.force_plot(*args, matplotlib=False, feature_names = plotly_df.columns[:-1])
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return html.Iframe(srcDoc=shap_html,
                       style={"width": "90%", "height": "200px", "border": 0})

    

if __name__ == '__main__':
    app.run_server(debug=True)