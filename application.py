import dash
import dash.dcc as dcc
import dash.html as html
import pandas as pd
from dash.dependencies import Input, Output
from dash import dash_table
import dash_daq as daq
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import requests as req




#App initialization, prevents of initial callbacks to have a tidy webpage
app = dash.Dash(__name__, prevent_initial_callbacks=True)

server = app.server

data = pd.read_csv("client_database.csv", index_col="SK_ID_CURR")

#import of the xgboost model and clients' data
model = xgb.XGBClassifier()
model.load_model("final_model_give_credit.json")

#defining the 20 first best features, e.g. the best fscores.
best_scores = pd.DataFrame.from_dict(model.get_booster().get_fscore(), orient='index').reset_index()
best_scores.columns = ['Features', 'Importance']
best_scores = best_scores.sort_values(by='Importance', ascending = False).iloc[:20,0].tolist()
best_scores = [x.rstrip("\n") for x in best_scores]

#isolating the 20 first best features for every clients
data_toplot = data.loc[:,best_scores]

#defining colors used for the webpage
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

#webpage layout

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

#callback to display a table of 20 firsts more important features of a selected client
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

#callback to display graphs of the distribution of the 20 first more important features and locating the selected client within this data
#(located by a white line) 
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

#a donut graph representing the score of our xgboost prediction  (propability*100)
@app.callback(
    Output('circle_proba', 'children'),
    Input(component_id='selected_client', component_property='value')
)
def circle_proba(selected_client):
    url = 'https://api-credit-openclassrooms.herokuapp.com/' + str(selected_client)
    resp = req.get(url)
    dico = resp.json()
    dico = dico.get("Proba_reimbursment")
    proba = dico.get(str(selected_client))
    palette = px.colors.qualitative.Vivid    
    score = round(proba*100,2)
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

#same as above, but with manual modifications of the values taken in account for calculating the score thanks to a editable table (ie "newtable")
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
        model = xgb.XGBClassifier()
        model.load_model("final_model_give_credit.json")
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
        graph = dcc.Graph(id = 'circle_plot', figure=fig)
        newtable = dash_table.DataTable(id = 'the_new_table',
            columns=(
            [{'id': p, 'name': p} for p in newdata]
        ),
        data=[newdata.to_dict(orient='list')],
        editable=False,
        fixed_rows={'headers': True},
        style_cell={
        'minWidth': 110, 'maxWidth': 110, 'width': 110
        })
        dccgraph = [graph, newtable]
    else:
        dccgraph = []
    return dccgraph


if __name__ == "__main__":
    app.run_server(debug=True)

