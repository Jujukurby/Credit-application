import dash
import dash.dcc as dcc
import dash.html as html
import pandas as pd
from dash.dependencies import Input, Output
import xgboost as xgb
import plotly.graph_objects as go

app = dash.Dash(__name__, prevent_initial_callbacks=True)

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



app.layout = html.Div(
    children=[
        html.H1(children="blahablah",),
        dcc.Dropdown(
        id='selected_client',
        options=[{'label':i, 'value':i} for i in data.index.unique()]
        ,

         placeholder="Select a client",
    ),
        html.P(
            id='graph-with-client',
            ),
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
    for col in  data.columns[0:-1]:
        fig = go.Figure()
        fig = fig.add_histogram(x = toplot[col])
        fig = fig.add_vline(client_data[col])
        output.append(dcc.Graph(id=col,figure=fig))
    return output


if __name__ == "__main__":
    app.run_server(debug=True)

