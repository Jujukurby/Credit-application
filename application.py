import dash
import dash.dcc as dcc
import dash.html as html
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

proba_test_client = model.predict_proba(data)

best_scores = pd.DataFrame.from_dict(model.get_booster().get_fscore(), orient='index').reset_index()
best_scores.columns = ['Features', 'Importance']
best_scores = best_scores.sort_values(by='Importance', ascending = False).iloc[:20,0].tolist()
best_scores = [x.rstrip("\n") for x in best_scores]
data = data.loc[:,best_scores]

data['Probability_of_reemboursement'] = proba_test_client[:,0]

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': 'black', 'text' : "white"},
    children=[
        html.H1(children="Credit : give it or not ?",),
        dcc.Dropdown(
        id='selected_client',
        options=[{'label':i, 'value':i} for i in data.index.unique()]
        ,

         placeholder="Select a client",
    ),
        html.P(
            dcc.Graph(id='circle_proba')
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
    client_data = data.loc[selected_client,:]
    toplot = data[data['Probability_of_reemboursement'] >= 0.7]
    output = []
    palette = px.colors.qualitative.G10
    palette = palette + palette
    i=0
    for col in  data.columns[0:-1]:
        fig = go.Figure()
        fig.add_histogram(x = toplot[col], marker=dict(color=palette[i]) )
        fig.add_vline(client_data[col])
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
    Output('circle_proba', 'figure'),
    Input(component_id='selected_client', component_property='value')
)
def circle_proba(selected_client):
    client_data = data.loc[selected_client,:]
    score = round(client_data['Probability_of_reemboursement']*100,2)
    values = [score, (100-score)]
    if score >= 70:
        score_color = 'green'
    elif score >= 50:
        score_color = 'yellow'
    else:
        score_color = 'red'
    fig = go.Figure(data=[go.Pie(values=values, hole = 0.7)])
    fig.update_layout( {
                'title' : 'Probabylity of reemboursment',
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                'color': colors['text'] }})
    fig.add_annotation(x= 0.5, y = 0.5,
                    text = str(score),
                    font = dict(size=20,family='Verdana', 
                                color= score_color),
                    showarrow = False)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)

