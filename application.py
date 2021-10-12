from operator import ne
import dash
import dash.dcc as dcc
from dash.exceptions import PreventUpdate
import dash.html as html
from dash.html.H2 import H2
from matplotlib.pyplot import figure, legend
import pandas as pd
from dash.dependencies import Input, Output
from dash import dash_table
import dash_daq as daq
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(__name__, prevent_initial_callbacks=True)

server = app.server

model = xgb.XGBClassifier()
model.load_model("final_model_give_credit.json")
data = pd.read_csv("client_database.csv", index_col="SK_ID_CURR")



best_scores = pd.DataFrame.from_dict(model.get_booster().get_fscore(), orient='index').reset_index()
best_scores.columns = ['Features', 'Importance']
best_scores = best_scores.sort_values(by='Importance', ascending = False).iloc[:20,0].tolist()
best_scores = [x.rstrip("\n") for x in best_scores]


data_toplot = data.loc[:,best_scores]

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': 'black'},
    children=[
        html.H1(children="Credit : give it or not ?",
                 style={'color' : colors['text']}),
        dcc.Dropdown(
        id='selected_client',
        options=[{'label':i, 'value':i} for i in data.index.unique()],
        placeholder="Select a client",
        ),
        html.H2(children="Change parameters for this client ?",
                 style={'color' : colors['text']}),
        daq.BooleanSwitch(id='change_table', on=False
        ),      
        html.P(
            id='new-table', 
            children = dash_table.DataTable(id = 'the_table',
            columns=(
        ),
        data=[],
        editable=False,
        fixed_rows={'headers': True},
        style_cell={
        'minWidth': 95, 'maxWidth': 95, 'width': 95
    })),
        html.P(
            id='circle_proba',
        ),
        html.P(
            id='circle_new_proba',
        ),
        html.P(
            id='graph-with-client',
            )
    ]
)


@app.callback(
    Output('new-table', 'children'),
    [Input('change_table', 'on'), Input('selected_client', 'value')]
)
def table_client(change_table, selected_client):
    data_client = data_toplot.loc[[selected_client]]
    if change_table:
        newtable = dash_table.DataTable(id = 'the_table',
            columns=(
            [{'id': p, 'name': p} for p in data_client]
        ),
        data=[data_client.to_dict(orient='list')],
        editable=True,
        fixed_rows={'headers': True},
        style_cell={
        'minWidth': 110, 'maxWidth': 110, 'width': 110
    })
    else:
        newtable = dash_table.DataTable(id = 'the_table',
            columns=(
            [{'id': p, 'name': p} for p in data_client]
        ),
        data=[data_client.to_dict(orient='list')],
        editable=False,
        fixed_rows={'headers': True},
        style_cell={
        'minWidth': 110, 'maxWidth': 110, 'width': 110
    })
    return newtable


@app.callback(
    Output('graph-with-client', 'children'),
    [Input(component_id='selected_client', component_property='value'), Input('change_table', 'on')]
)
def update_output(selected_client, table):
    output = []
    if table == False:
        client_data = data_toplot.loc[selected_client,:]
        palette = px.colors.qualitative.G10
        palette = palette + palette
        
        for i, col in  enumerate(data_toplot.columns[0:-1]):
            fig = go.Figure()
            fig.add_histogram(x = data_toplot[col], marker_color=palette[i])
            fig.add_vline(client_data[col], line_color="white")
            fig.update_layout( {
                    'title' : col,
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                    'color': colors['text'] 
                    }
                    })
            output.append(dcc.Graph(id=col,figure=fig))
    return output

@app.callback(
    Output('circle_proba', 'children'),
    Input(component_id='selected_client', component_property='value')
)
def circle_proba(selected_client):
    palette = px.colors.qualitative.Vivid
    client_data = data.loc[[selected_client]]
    proba = model.predict_proba(client_data)
    score = round(proba[0][0]*100,2)
    values = [score, (100-score)]
    if score >= 70:
        score_color = palette[3]
    elif score >= 50:
        score_color = palette[0]
    else:
        score_color = palette[9]
    fig = go.Figure(data=[go.Pie(values=values, hole = 0.7)])
    fig.update_layout( {
                'title' : 'Probability of reimbursement',
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                'color': colors['text'] },
                'showlegend':False})

    fig.update_traces(hoverinfo='label+percent', textinfo='none', textfont_size=20,
                  marker=dict(colors=[score_color, 'grey']))
    fig.add_annotation(x= 0.5, y = 0.5,
                    text = str(score),
                    font = dict(size=20,family='Verdana', 
                                color= score_color),
                    showarrow = False)
    dccgraph = dcc.Graph(id = 'circle_plot', figure=fig)
    return dccgraph

@app.callback(
    Output('circle_new_proba', 'children'),
    Input(component_id='selected_client', component_property='value'),
    Input('change_table', 'on'),
    Input('the_table', 'data'),
    Input('the_table', 'columns')
)
def circle_new_proba(selected_client, table, rows, columns):
    if table:
        palette = px.colors.qualitative.Vivid
        client_data = data.loc[[selected_client]]
        newdata = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        newdata['SK_ID_UNIQ'] = selected_client
        newdata.set_index('SK_ID_UNIQ', drop=True, inplace=True)
        for col in newdata.columns:
            client_data.loc[selected_client,col] =  newdata.loc[selected_client,col]
        proba = model.predict_proba(client_data)
        score = round(proba[0][0]*100,2)
        values = [score, (100-score)]
        if score >= 70:
            score_color = palette[3]
        elif score >= 50:
            score_color = palette[0]
        else:
            score_color = palette[9]
        fig = go.Figure(data=[go.Pie(values=values, hole = 0.7)])
        fig.update_layout( {
                    'title' : 'New probability of reimbursement',
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                    'color': colors['text'] },
                    'showlegend':False})

        fig.update_traces(hoverinfo='label+percent', textinfo='none', textfont_size=20,
                    marker=dict(colors=[score_color, 'grey']))
        fig.add_annotation(x= 0.5, y = 0.5,
                        text = str(score),
                        font = dict(size=20,family='Verdana', 
                                    color= score_color),
                        showarrow = False)
        dccgraph = dcc.Graph(id = 'circle_plot', figure=fig)
    else:
        dccgraph = []
    return dccgraph


if __name__ == "__main__":
    app.run_server(debug=True)

