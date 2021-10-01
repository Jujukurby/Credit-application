import dash
import dash.dcc as dcc
import dash.html as html
import pandas as pd
from dash.dependencies import Input, Output


data = pd.read_csv("best_model_probas.csv")

app = dash.Dash(__name__)
output = []
for col in  data.columns[1:-1]:
     output.append(dcc.Graph(id=col,figure={"data": [
                        {
                            "x": data[data['Probability_of_reemboursement'] >= 0.7][col],
                            "type": "histogram",
                        },
                    ],
                    "layout": {"title": col}, }))



app.layout = html.Div(
    children=[
        html.H1(children="blahablah",),
        dcc.Dropdown(
        id='selected-client',
        options=[{'label':i, 'value':i} for i in data['SK_ID_CURR'].unique()]
        ,

         placeholder="Select a client",
    ),
        html.P(
            children=output,
            ),
    ]
)

@app.callback(
    Output(component_id='client_data', component_property='children'),
    Input(component_id='selected_client', component_property='value')
)

def update_output_div(selected_client):
    client_data = data[data['SK_ID_CURR'] == selected_client]
    return client_data


if __name__ == "__main__":
    app.run_server(debug=True)

