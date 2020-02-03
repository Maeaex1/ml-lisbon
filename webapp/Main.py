import pandas as pd
import numpy as np

import json
import math
from random import gauss
from random import seed

seed(1)

import prepareDataSet as preper
import models as ml

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State

import plotly.graph_objects as go

mapbox_access_token = open(".mapbox_token").read()

file_name = 'cleanDataSet_ref.csv'
geo_file = 'districts_lisbon.geojson'
df_data = pd.read_csv(file_name)
df_data = df_data.sample(frac=1)

#Get rid of possible time dependencies because of crawling order

df = preper.create_features(df_data)
df = preper.impute_missing(df)
df = preper.remove_outliers(df)
coords = df.filter(['altitude', 'longitude'])# save short in other dataframe
noise1 = 0.0007 * pd.Series([gauss(0.0, 1.0) for i in range(len(coords))])
noise2 = 0.0007 * pd.Series([gauss(0.0, 1.0) for i in range(len(coords))])
coords['longitude'] = coords['longitude'] + noise1
coords['altitude'] = coords['altitude'] + noise2

df = df.drop(columns=['altitude', 'longitude', 'Unnamed: 0'], axis=1)

#may drop 'area' + comment line in prepareDataset
categorial_features = df.dtypes[df.dtypes == "object"].index.to_list()
categorial_features.extend(['no_rooms', 'no_bathroom'])
numerical_features = df.dtypes[df.dtypes != "object"].index.to_list()
district_select = list(df.district.unique())
exclude_type = list(df.estate_type.unique())

df = pd.concat([df, coords] ,axis=1)

with open(geo_file, encoding='utf-8') as resp:
    districts = json.load(resp)

external_css = ['venv/assets/style.css'] ## Load my table grid css
app = dash.Dash(__name__,
                external_stylesheets=external_css)

server = app.server

app.layout = html.Div([
    html.Header([
        html.H1('Max Analytics')
    ]),
    html.Div(id='intermediate-value', style={'display':'none'}),
    html.Div([
        html.H1('Lisbon Housing Market', style={'text-align':'center'}),
        html.Div([ ### Here starts Table
            html.Div([
                html.P('Select Target Variable'),
                dcc.Dropdown(id='output-var',
                             options=[{'label': 'Price per m²', 'value': 'sqm_price_usage'},
                                      {'label': 'Total Price', 'value': 'price'}],
                             value='price'
                             ),
                html.P('Select Group Variable'),
                dcc.Dropdown(id='group-by',
                             options=[{'label': 'District', 'value': 'district'},
                                      {'label': 'Area', 'value': 'area'},
                                      {'label': 'Condition', 'value': 'condition'},
                                      {'label': 'Estate Type', 'value': 'estate_type'},
                                      {'label': 'Energy Efficiency', 'value': 'energy_efficiency'},
                                      {'label': 'Number of Rooms', 'value': 'no_rooms'},
                                      {'label': 'Number of Bathrooms', 'value': 'no_bathroom'},
                                      ],
                             value='district'
                             ),
                html.P('Filter by district'),
                dcc.Dropdown(
                    id='district-select',
                    options=[{'label': i, 'value': i} for i in district_select],
                    value=district_select,
                    multi=True,
                ),
            ], className='panel'),
            html.Div([
                html.Div([
                    html.H2(html.Div(id='num-properties', className='panel-value')),
                    html.H4('Number of properties', className='description'),
                ], className='panel1'),
                html.Div([
                    html.H2(html.Div(id='avg-price', className='panel-value')),
                    html.H4('Average property price', className='description')
                ], className='panel2'),
                html.Div([
                    html.H2(html.Div(id='avg-sqm', className='panel-value')),
                    html.H4('Average property size', className='description')
                ], className='panel3'),
                html.Div([
                    html.H2(html.Div(id='avg-sqm-price', className='panel-value')),
                    html.H4('Average price/m²', className='description')
                ], className='panel4'),
            ], className='summary-div'),
            html.Div([
                html.Div([
                    html.P('Filter by sqm'),
                    html.Div([
                        dcc.RangeSlider(
                            id='sqm-slider',
                            min=math.floor(df.area_usage.min()),
                            max=math.ceil(df.area_usage.max()),
                            step=1,
                            allowCross=False,
                            value=[math.floor(df.area_usage.min()), math.ceil(df.area_usage.max())]),
                        html.Div(id='output-sqm')
                    ], style={'margin':'5px 5px 5px 5px'}),
                ], className='filter1'),
                html.Div([
                    html.P('Exclude Property Types'),
                    html.Div([
                        dcc.Dropdown(
                            id='ex-estate-type',
                            options=[{'label': i, 'value': i} for i in exclude_type],
                            value=['Garagem', 'Hotel', 'Restaurante'],
                            multi=True,
                        ),
                    ]),
                ]),
            ], className='filter-div'),
            html.Div([
                dcc.Graph(id='graph1')
            ], className='graph1'),
            html.Div([
                dcc.Graph(id='graph2')
            ], className='graph2'),
            html.Div([
                dcc.Graph(id='graph3'),
                html.Div([
                    dcc.Dropdown(id='xaxis-value',
                                 options=[{'label': 'Total m²', 'value': 'area_total'},
                                          {'label': 'Habital m²', 'value': 'area_usage'},
                                          {'label': 'Price per m² (Total)', 'value': 'sqm_price_total'},
                                          {'label': 'Price per m² (Usage)', 'value': 'sqm_price_usage'},
                                          {'label': 'Avg. Room Size', 'value': 'avg_room_size'},
                                          ],
                                 value='area_total'
                                 ),
                ], style={'width':'40%', 'text-align':'center', 'margin':'0 auto'})
            ], className='graph3'),
        ], className='grid-container'),
        html.Div([
            html.Div([
                dcc.Markdown('''

            ## Predicting House Prices with Machine Learning
            Based on your data selection we are performing house price predictions.
            **Variables that are derived from prices e.g. price per sqm are excluded.**
            Keep in mind - In general the more observations the better for training purposes.
            
            ### Data Set
            Based on your selection we take 90% of data for training and 10% for testing.
            All Data used for this analysis was crawled (extracted) from [ERA Imobiliária](https://www.era.pt/imoveis/comprar/?q=comprar%20Lisboa%).
            The data set will be updated every month and maybe extended with other information (e.g. Airbnb or other real estate platforms etc.). 
                        
            You can request the data set via the contact formular on my [website](www.max-analytics.de).
            You can find the used models for predictions in my [github repository].
            
            ### Outlier Dectection
            We will skip outlier detection as you can assess this by your own (using the charts above).
            For visualization purposes we excluded already the bottom & top 1-percentile for price & total m².
            For those with a little bit of money or with 'Monopoly' ambitions you can buy 
            [this planned hotel](https://www.era.pt/imoveis/predio-lisboa-marvila_pt_1025482) for 19.5 Mio € in beautiful Marvila.
            
            ### Missing Data
            Data imputation has also been made for missing values already. For continuous variables we imputed median values, for categorials "None".
            
            ## Model Methodology
            
            We use the following models:
            
            1. XGB's Gradient Boosting - Regression Trees
            2. Ridge Regression
            3. Elastic Net
                       
            ### Data Transformation + Model Training
            We use Sklearn's Pipeline function + GridSearchCV to perform model selection.
            
            As depending on the data choice we face heavy outliers in prices, so we use RobustScaler for continous variables. 
            We do not unskew data nor use log prices to cure from heteroscedasticity. I leave this up for another project.
            Categorical Features are transformed to Dummies using LabelEncoder() & pd.get_dummies function

            Then we perform 5-fold crossvalidation to train & test our models with different hyperparameters.
            * Regularization parameter "alpha" for Lasso & Elastic Net for values (0.5, 1, 1.5)
            * Pseudo-Regularization parameter "gamma" for Gradient Boosting for values (0, 1), "0" means no regularization + tree depth
            between (8, 10)
            * As we use a Scaler we do not normalize our data 
            
            ### Click the Button below to run the models (may take a few seconds)
                     
            '''
                             )
            ], className='markdown-area'),
            html.Div([
                html.Button('Start Models', id='button'),
                html.Div(id='output-score', className='output-score'),
                html.Div([
                    dcc.Graph(id='map-final')
                ]),
                dcc.Markdown(
                    '''
                    ### XGB's Gradient Boost outperforms on Absolute Median Error loss function by far.
                    
                    However, due to the lack of observations and a relatively small amount of features some predictions
                    come with relatively large outliers. 
                    
                    **The markers on the map do not show the actual location of the properties**
                    The data craweld from Era does only contain the centroid coordinates of the districts.
                    We added some Gaussian noise to the coordinates to make it look better. 
                    
                    I am currently developing a Neural Network in PyTorch to outperform XGB's Gradient Boost's prediction accuarcy.
                    
                    So stay tuned!
                    
                    
                    ## List of variables:
                    
                    ### Categorical
                    
                    * Number of Bathrooms
                    * Number of Rooms
                    * District (25 districts of Lisbon + 'None')
                    * Estate Type (21 different categories)
                    * Condition (used, new, in construction, planned, renovated, NA)
                    * Energy efficiency (A to F + 'None')
                    
                    ### Continuous
                    
                    * Price (target variable)
                    * Area Usage (habital sqm)
                    * Area Total (Area Usage + Outdoor Area)
                    
                    '''
                )
            ], className='markdown-area')
        ],className='ml-area')
    ], className='container'),
])

@app.callback(
    Output('output-sqm', 'children'),
    [Input('sqm-slider', 'value')]
)
def update_output(value):
    v1 = value[0]
    v2 = value[1]
    return 'Properties between {}m² and {}m² selected '.format(v1, v2)

@app.callback(
    Output('intermediate-value', 'children'),
    [Input('district-select', 'value'),
     Input('sqm-slider', 'value'),
     Input('ex-estate-type', 'value')],
)

def prepare_data(district_select, sqm, ex_type):
    df_filter = df[df['district'].isin(district_select)]
    df_filter = df_filter[(df_filter.area_usage >= sqm[0]) & (df_filter.area_usage <= sqm[1])]
    df_filter = df_filter[~df_filter['estate_type'].isin(ex_type)]

    return df_filter.to_json(date_format='iso', orient='split')


@app.callback([
    Output('num-properties', 'children'),
    Output('avg-price', 'children'),
    Output('avg-sqm', 'children'),
    Output('avg-sqm-price', 'children')],
    [Input('intermediate-value', 'children')],
)

def calc_overview(df_filter):

    df_filter = pd.read_json(df_filter, orient='split')

    num_properties = '{}'.format(len(df_filter))
    avg_price = '{:,.0f} €'.format(df_filter.price.mean())
    avg_sqm = '{:,.2f} m²'.format(df_filter.area_usage.mean())
    avg_sqm_price = '{:,.2f} €/m²'.format(df_filter.sqm_price_usage.mean())

    return num_properties, avg_price, avg_sqm, avg_sqm_price


@app.callback(
    Output('graph1', 'figure'),
    [Input('intermediate-value', 'children'),
     Input('group-by', 'value'),
     Input('output-var', 'value')]
)

def update_graph1(df_filter, selection, output):
    df_filter = pd.read_json(df_filter, orient='split')

    N = df_filter[selection].unique()
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(N))]

    traces = []
    for idx, select in enumerate(N):
        traces.append(go.Box(y=df_filter[df_filter[selection] == select][output], name=str(select), marker={"size": 4}, marker_color=c[idx]))

    layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       margin={"r": 0, "t": 16 },
                       xaxis_type='category',
                       yaxis_title=output)
    return {"data": traces, "layout": layout}


@app.callback(
    Output('graph2', 'figure'),
    [Input('intermediate-value', 'children'),
     Input('output-var', 'value')]
)

def update_graph2(df_filter, selected):
    df_filter = pd.read_json(df_filter, orient='split')
    df_grouped = df_filter.groupby(['district']).median()

    fig = [go.Choroplethmapbox(
        geojson=districts, locations=df_grouped.index, z=df_grouped[selected],
        featureidkey="properties.NOME", colorscale="Viridis", zmin=df_grouped[selected].min(), zmax=df_grouped[selected].max(),
        marker_opacity=0.5, marker_line_width=0)]

    layout = go.Layout(mapbox_style="carto-positron",
                      mapbox_zoom=11, mapbox_center={"lat": 38.734338, "lon": -9.1591},
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       margin={"r": 0, "t": 0, "l": 0, "b": 0},
                       )

    return {"data": fig, "layout": layout}


@app.callback(
    Output('graph3', 'figure'),
    [Input('intermediate-value', 'children'),
     Input('xaxis-value', 'value'),
     Input('output-var', 'value')]
)

def update_graph3(df_filter, xaxis, output):
    df_filter = pd.read_json(df_filter, orient='split')

    fig = [go.Scatter(
        x=df_filter[xaxis], y=df_filter[output], mode='markers', marker_color='hsl(195.0,50%,50%)'
    )]

    layout = go.Layout(
        xaxis_title=xaxis,
        yaxis_title=output
    )

    return {'data': fig, 'layout': layout}

@app.callback(
    [Output('output-score', 'children'),
    Output('map-final', 'figure')],
    [Input('button', 'n_clicks')],
    [State('intermediate-value', 'children')],

)

def model_valuation(n_clicks, df_filter):
    df_filter = pd.read_json(df_filter, orient='split')
    data = ml.encodeLabels(df_filter)
    X_train, X_test, y_train, y_test = ml.crete_data_set_ML(data)


    score_xgb, pred_xgb, est_xgb = ml.init_XGB(X_train, X_test, y_train, y_test)
    score_ridge, pred_ridge, est_ridge = ml.init_Ridge(X_train, X_test, y_train, y_test)
    score_ElNet, pred_ElNet, est_elnet = ml.init_ElNet(X_train, X_test, y_train, y_test)

    useful_info = df_filter.tail(len(pred_xgb))
    predictions = pd.DataFrame(np.concatenate((pred_xgb.reshape(-1,1), pred_ridge.reshape(-1,1), pred_ElNet.reshape(-1,1)), axis=1), columns=['pred_XGB', 'pred_Ridge', 'pred_ElNet'])
    predictions = predictions.round(0)
    all_info = pd.concat([predictions.reset_index(drop=True), useful_info.reset_index(drop=True)], axis=1)
    all_info['text'] = 'Price: ' + all_info['price'].astype(str) + ', XGB: ' + all_info['pred_XGB'].astype(str) + ', Ridge: ' + all_info['pred_Ridge'].astype(str) + ', Elastic Net: ' + all_info['pred_ElNet'].astype(str)
    overall_valuation = "Median Absolute Errors: XBG = {:,.0f} €, Ridge = {:,.0f} €, Elastic Net {:,.0f} €".format(-score_xgb, -score_ridge, -score_ElNet)

    cut_df = all_info.head(30)

    fig = [go.Scattermapbox(
        lat=cut_df['longitude'].astype(str).to_list(),
        lon=cut_df['altitude'].astype(str).to_list(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=14
        ),
        text=cut_df['text'].to_list(),
    )]

    layout = go.Layout(
        title='Predictions vs Actual for 30 Properties',
        hovermode='closest',
        margin={"r": 0, "t": 35, "l": 0, "b": 0},
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=38.734338,
                lon=-9.1591
            ),
            pitch=0,
            zoom=11,
        ))


    return overall_valuation, {'data': fig, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)