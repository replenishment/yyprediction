from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
import pandas as pd
import numpy as np
import sqlite3
# from sklearn import ensemble
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import xgboost
from xgboost import XGBRegressor

# Load the training data
df = pd.read_excel('./Training Data - 111822.xlsx')

y = df['Avg Net YY'].values
X_ori = df.iloc[:,53:68].values

# SubGroup for specific prediction
z = df.iloc[:,1].values

# Convert SubGroup to One Hot Encoder
brand_ohe = OneHotEncoder()
z = brand_ohe.fit_transform(z.reshape(-1,1)).toarray()

# Add SubGroup (OneHotEncoded) to features column
X = np.c_[z,X_ori]

## Using Gradient Boosting Model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# params = {
#             "n_estimators": 500,
#             "max_depth": 4,
#             "min_samples_split": 5,
#             "learning_rate": 0.01,
#             "loss": "squared_error",
#         }
# reg = ensemble.GradientBoostingRegressor(**params)
# reg.fit(X_train, y_train)
# mse = mean_squared_error(y_test, reg.predict(X_test))

# Using XGBoosting Model
model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model.fit(X, y)

# Connect to Sqlite Database
conn = sqlite3.connect('‪yy predict.db')
c = conn.cursor()
statement = '''SELECT * FROM prediction'''
c.execute(statement)
global df_test
df_test = pd.read_sql_query(statement, conn)
conn.commit()
conn.close()

app = Dash(__name__)

app.layout = html.Div([
    # Dropdown List to select a brand name
    html.Label('Select your Brand'),
    dcc.Dropdown(
        df['Brand'].unique(),
        id='brand-dropdown',
        placeholder="Others",
        value='Others'
    ),

    html.Br(),

    # Radio Button to select dress shirt or sports shirt
    dcc.RadioItems(
        id='type-radio-item',
        options=[
            {'label': 'Dress Shirt', 'value': 'dress'},
            {'label': 'Sports Shirt', 'value': 'sport'}
        ],
        value='dress'
    ),

    html.Br(),

    # Radio Button to select Solid,Stripe or Check
    dcc.RadioItems(
        id='pattern-radio-item',
        options=[
            {'label': 'Solid', 'value': 'solid'},
            {'label': 'Stripe', 'value': 'stripe'},
            {'label': 'Check', 'value': 'check'}
        ],
        value='solid'
    ),

    html.Br(),

    # Radio Button to select one-way-match or two-way-match
    dcc.RadioItems(
        id='match-radio-item',
        options=[
            {'label': 'One Way Match', 'value': 'owm'},
            {'label': 'Two way Match', 'value': 'twm'}
        ],
        value='twm'
    ),

    html.Br(),

    # Radio Button to select Short Sleeve or Long sleeve
    dcc.RadioItems(
        id='sleeve-radio-item',
        options=[
            {'label': 'Short Sleeve', 'value': 'ss'},
            {'label': 'Long Sleeve', 'value': 'ls'}
        ],
        value='ls'
    ),

    html.Br(),

    # Input for plan cut qty
    html.Label('Plan Cut Qty : '),
    dcc.Input(
        id='pcq-input',
        placeholder='Input plan cut qty in yard',
        type='number',
        min=0,
        max=100000,
        value='500'
    ),

    html.Br(),
    html.Br(),

    # Input for repeat X
    html.Label('Repeat X : '),
    dcc.Input(
        id='rx-input',
        placeholder='Input repeat X :',
        type='number',
        min=0,
        max=5,
        value='0'
    ),

    # Input for repeat Y
    html.Label('  Repeat Y : '),
    dcc.Input(
        id='ry-input',
        placeholder='Input repeat y :',
        type='number',
        min=0,
        max=5,
        value='0'
    ),

    html.Br(),
    html.Br(),

    # Input for average neck size
    html.Label('Average neck size :'),
    dcc.Slider(30, 45,
               value=33,
               marks=None,
               id='ans-slider',
               tooltip={"placement": "bottom", "always_visible": True}
               ),

    # Radio Button to select single cuff or double cuff
    dcc.RadioItems(
        id='cuff-radio-item',
        options=[
            {'label': 'Single Cuff', 'value': 'sc'},
            {'label': 'Double Cuff', 'value': 'dc'}
        ],
        value='sc'
    ),

    html.Br(),

    # Input for marker width
    html.Label('Marker width :'),
    dcc.Slider(40, 70,
               value=60,
               marks=None,
               id='mw-slider',
               tooltip={"placement": "bottom", "always_visible": True}
               ),

    # Radio Button to select fit1,fit2 or fit3
    dcc.RadioItems(
        id='fit-radio-item',
        options=[
            {'label': 'classic fit', 'value': 'cf'},
            {'label': 'slim fit', 'value': 'sf'},
            {'label': 'extra slim fit', 'value': 'esf'}
        ],
        value='cf'
    ),

    html.Br(),

    # Input for average neck size
    html.Label('Actual CMD yy : '),
    dcc.Input(
        id='cmdyy-input',
        placeholder='Input Actual CMD yy :',
        type='number',
        min=0,
        max=3,
        value='0'
    ),

    html.Br(),
    html.Br(),

    html.Label('Any of special features in your style below :'),
    dcc.Checklist(
        id='feature-checklist',
        options=[
            {'label': 'No pocket', 'value': 'npt'},
            {'label': 'Double pocket', 'value': 'dpt'},
            {'label': 'Pocket with flap', 'value': 'pte'},
            {'label': 'Emboss collar', 'value': 'ptf'},
            {'label': 'Mandarin collar', 'value': 'mcr'},
            {'label': 'Engineered pattern', 'value': 'ep'},
        ],
        value=[]
    ),

    html.Br(),

    # Input for style number
    html.Label('Input style number for reference : '),
    dcc.Input(
        id='style-input',
        value='',
        type='text',
    ),

    html.Br(),
    html.Br(),

    dcc.Textarea(
        id='remark-textarea',
        placeholder='Any remark',
        value='',
        style={'width': '50%'}
    ),

    html.Br(),
    html.Br(),

    # Button to submit all input at once
    html.Button('Press Button to Predict yy',
                id='submit-val',
                n_clicks=0
                ),

    html.Br(),
    html.Br(),

    # Container for showing the current yy prediction result
    html.Div(id='my-output'),

    html.Br(),
    html.Br(),

    # Button to save to the YY prediction result to database
    html.Button("Save", id="btn-save"),

    html.Br(),

    html.Div(id='save-output'),

    html.Br(),

    # Button to save the YY prediction result
    html.Button('Show Table',
                id='submit-save',
                n_clicks=0
                ),

    html.Br(),
    html.Br(),

    # Container for showing all input from user and the saved yy prediction result

    dash_table.DataTable(id='datatable-output',
                         columns=[
                             {"name": i, "id": i, "deletable": True, "selectable": True} for i in df_test.columns
                         ],
                         data=df_test.to_dict('records'),
                         filter_action="native",
                         sort_action="native",
                         sort_mode="multi",
                         column_selectable="single",
                         row_selectable="multi",
                         row_deletable=True,
                         selected_columns=[],
                         selected_rows=[],
                         page_action="native",
                         page_current=0,
                         page_size=10,

                         ),

    html.Br(),
    html.Br(),

    # Button to download to CSV of the YY prediction result
    html.Button("Download CSV", id="btn-csv"),
    dcc.Download(id="download-dataframe-csv")

])


@app.callback(
    Output('my-output', 'children'),
    Input('submit-val', 'n_clicks'),
    State('brand-dropdown', 'value'),
    State('type-radio-item', 'value'),
    State('pattern-radio-item', 'value'),
    State('match-radio-item', 'value'),
    State('sleeve-radio-item', 'value'),
    State('pcq-input', 'value'),
    State('rx-input', 'value'),
    State('ry-input', 'value'),
    State('ans-slider', 'value'),
    State('cuff-radio-item', 'value'),
    State('mw-slider', 'value'),
    State('fit-radio-item', 'value'),
    State('cmdyy-input', 'value'),
    State('feature-checklist', 'value'),
    State('style-input', 'value'),
    State('remark-textarea', 'value'),
    prevent_initial_call=True
)
def update_data(n_clicks, selected_brand, selected_type, selected_pattern, selected_match, selected_sleeve,
                selected_pcq,
                selected_rx, selected_ry, selected_ans, selected_cuff, selected_mw, selected_fit, selected_cmdyy,
                selected_feature, selected_style, selected_remark):
    global brand_subgroup
    brand_subgroup = df['Brand code for calculation'].unique()
    no_of_subgroup = len(brand_subgroup)
    bg_array = []
    for i in range(0, no_of_subgroup):
        t = 0
        bg_array.append(t)

    # Determine the brand subgroup for regression
    if selected_brand == 'CST' and selected_cuff == 'sc':
        bg_array[0] = 1
    elif selected_brand == 'CST' and selected_cuff == 'dc':
        bg_array[1] = 1
    elif selected_brand == 'PAU' and selected_type == 'dress':
        bg_array[2] = 1
    elif selected_brand == 'PAU' and selected_type == 'sport':
        bg_array[12] = 1
    elif selected_brand == 'BBR' and selected_pattern == 'solid':
        bg_array[4] = 1
    elif selected_brand == 'BBR' and selected_pattern == 'stripe':
        bg_array[3] = 1
    elif selected_brand == 'BBR' and selected_pattern == 'check':
        bg_array[5] = 1
    elif selected_brand == 'DIL':
        bg_array[6] = 1
    elif selected_brand == 'DIL-Cremieux':
        bg_array[7] = 1
    else:
        bg_array[9] = 1

    # Change selected_feature from a list to string
    feature_string = ' ,'.join(map(str, selected_feature))

    if selected_pattern == 'solid':
        solid_predictor = 1
        stripe_predictor = 0
        check_predictor = 0
    elif selected_pattern == 'stripe':
        solid_predictor = 0
        stripe_predictor = 1
        check_predictor = 0
    else:
        solid_predictor = 0
        stripe_predictor = 0
        check_predictor = 1

    if selected_match == 'owm':
        owm_predictor = 1
        twm_predictor = 0
    else:
        owm_predictor = 0
        twm_predictor = 1

    if selected_sleeve == 'ss':
        ls_predictor = 0
    else:
        ls_predictor = 1

    pcq_predictor = selected_pcq
    rx_predictor = selected_rx
    ry_predictor = selected_ry
    ans_predictor = selected_ans

    if selected_cuff == 'sc':
        dc_predictor = 0
    else:
        dc_predictor = 1

    mw_predictor = selected_mw

    if selected_fit == 'cf':
        cf_predictor = 1
        sf_predictor = 0
        esf_predictor = 0
    elif selected_fit == 'sf':
        cf_predictor = 0
        sf_predictor = 1
        esf_predictor = 0
    else:
        cf_predictor = 0
        sf_predictor = 0
        esf_predictor = 1

    # Create a list of predictors for prediction
    list_X = (bg_array[0], bg_array[1], bg_array[2], bg_array[3], bg_array[4], bg_array[5],
              bg_array[6], bg_array[7], bg_array[8],
              bg_array[9], bg_array[10], bg_array[11], bg_array[12], solid_predictor, stripe_predictor, check_predictor,
              owm_predictor, twm_predictor, ls_predictor,
              pcq_predictor, rx_predictor, ry_predictor, ans_predictor, dc_predictor, mw_predictor,
              cf_predictor, sf_predictor, esf_predictor)

    # Predict the yy
    X_input = np.array(list_X, dtype=object).reshape(1, -1)
    y_output = model.predict(X_input)
    global y_output_float
    y_output_float = float("".join(map(str, y_output)))

    # Create a dataframe for download
    date = dt.today()
    list_not_predictor = (
    y_output_float, selected_cmdyy, selected_brand, feature_string, selected_style, selected_remark, date)
    list_download = list_X + list_not_predictor
    global df_download
    df_download = pd.DataFrame(list_download)

    return 'Predicted yy is {}'.format(y_output)


@app.callback(
    Output("save-output", "children"),
    Input("btn-save", "n_clicks"),
    prevent_initial_call=True
)
# Download the user input and predicted yy
def func(n_clicks):
    sql_header = ['subgroup_1', 'subgroup_2', 'subgroup_3', 'subgroup_4', 'subgroup_5', 'subgroup_6',
                  'subgroup_7', 'subgroup_8', 'subgroup_9', 'subgroup_10', 'subgroup_11', 'subgroup_12',
                  'subgroup_13', 'solid', 'stripe', 'chk', 'One_Way_Match', 'Two_Way_Match', 'Long_Sleeve',
                  'Plan_Cut_Qty',
                  'Repeat_X', 'Repeat_Y', 'Average_Neck_Size', 'Double_Cuff', 'Marker_Width', 'Regular_Fit',
                  'Slim_Fit', 'Extra_Slim_Fit', 'Predicted_yy', 'Actual_CMD_yy', 'Brand', 'Feature', 'Style',
                  'Remark_Text', 'Date']

    global sql_download
    sql_download = pd.DataFrame(np.array(df_download).reshape(1, -1), columns=sql_header)
    conn = sqlite3.connect('‪yy predict.db')
    c = conn.cursor()
    sql_download.to_sql('prediction', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()

    return 'Saved'


@app.callback(
    Output("datatable-output", "data"),
    Input("submit-save", "n_clicks"),
    prevent_initial_call=True
)
def func(n_clicks):
    # Show DataFrame in Dash Table
    conn = sqlite3.connect('‪yy predict.db')
    c = conn.cursor()
    statement = '''SELECT * FROM prediction'''
    c.execute(statement)
    # global df_test
    df_test = pd.read_sql_query(statement, conn)
    conn.commit()
    conn.close()
    return df_test.to_dict('records')


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-csv", "n_clicks"),
    prevent_initial_call=True
)
# Download the user input and predicted yy
def func(n_clicks):
    return dcc.send_data_frame(df_test.to_csv, "mydf.csv")


if __name__ == '__main__':
    app.run_server(port=4050)