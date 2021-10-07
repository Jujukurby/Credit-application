import dash
import dash.dcc as dcc
import dash.html as html
from matplotlib.pyplot import figure, legend
import pandas as pd
from dash.dependencies import Input, Output
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
        html.P(
            id='circle_proba',
            ),
        html.P(
            id='graph-with-client',
            )
    ]
)

@app.callback(
    Output('graph-with-client', 'children'),
    Input(component_id='selected_client', component_property='value')
)
def update_output(selected_client):
    client_data = data_toplot.loc[selected_client,:]
    output = []
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


if __name__ == "__main__":
    app.run_server(debug=True)

