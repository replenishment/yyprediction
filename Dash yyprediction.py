from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State


from datetime import datetime as dt
import pandas as pd
import numpy as np

import sqlite3

from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import xgboost
from xgboost import XGBRegressor


df = pd.read_excel('C:/Users/chans/YY Model/Training Data - 111822.xlsx')

y = df['Avg Net YY'].values
X_ori = df.iloc[:,53:68].values

# SubGroup for specific prediction
z = df.iloc[:,1].values

# Convert SubGroup to One Hot Encoder
brand_ohe = OneHotEncoder()
z = brand_ohe.fit_transform(z.reshape(-1,1)).toarray()

# Add SubGroup (OneHotEncoded) to features colunm
X = np.c_[z,X_ori]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# model = XGBRegressor()
model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model.fit(X, y)


# Predict training data and save to Excel
# predicted_yy = model.predict(X_ori)
# Result_Table = df
# Result_Table['Predicted_yy'] = predicted_yy

# df.to_excel(r'C:\users\chans\YY Model\XGBoost Result.xlsx', index=False)

app = Dash(__name__)

app.layout = html.Div([
    # Dropdown List to select a brand name
    html.Label('Select your Brand'),
    dcc.Dropdown(
        df['Brand'].unique(),
        id='brand-dropdown',
        value='Others'
    ),

    html.Br(),

    # Radio Button to select Solid,Stripe or Check
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

    html.Br(),
    html.Br(),

    # Input for repeat Y
    html.Label('Repeat Y : '),
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

    html.Br(),

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

    #     # Input for marker width
    #     html.Label('Marker width :'),
    #     dcc.Input(
    #         id='mw-input',
    #         placeholder='Input plan marker width',
    #         type='number',
    #         value='60'
    #     ),

    html.Br(),

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

    html.Label('Any of below special features in your style'),
    dcc.Checklist(
        options=[
            {'label': 'No pocket', 'value': 'NPT'},
            {'label': 'Double pocket', 'value': 'DPT'},
            {'label': 'Pocket with flap', 'value': 'PTF'},
            {'label': 'Mandarin collar', 'value': 'MCR'},
            {'label': 'Engineered pattern', 'value': 'EPF'},
            {'label': 'None of them', 'value': 'NUL'}
        ],
        value=['NUL']
    ),

    html.Br(),

    # Input for style number
    html.Label('Input style number for reference : '),
    dcc.Input(
        id='style-input',
        placeholder='Input style number',
        value='Input your style',
        type='text',
    ),

    html.Br(),

    dcc.Textarea(
        id='remark-textarea',
        placeholder='Input your comment',
        value='Input your comment',
        style={'width': '50%'}
    ),

    html.Br(),
    html.Br(),

    # Button to submit all input at once
    html.Button('Submit',
                id='submit-val',
                n_clicks=0
                ),

    html.Br(),
    html.Br(),

    # Container for all input from user
    html.Div(id='my-output'),

    html.Br(),
    html.Br(),

    html.Button("Download CSV", id="btn_csv"),
    dcc.Download(id="download-dataframe-csv")

])


@app.callback(
    Output('my-output', 'children'),
    Input('submit-val', 'n_clicks'),
    State('brand-dropdown', 'value'),
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
    State('style-input', 'value'),
    State('remark-textarea', 'value'),
    prevent_initial_call=True
)
def update_data(n_clicks, selected_brand, selected_pattern, selected_match, selected_sleeve, selected_pcq,
                selected_rx, selected_ry, selected_ans, selected_cuff, selected_mw, selected_fit, selected_cmdyy,
                selected_style, selected_remark):
    global brand_subgroup
    brand_subgroup = df['Brand code for calculation'].unique()
    no_of_subgroup = len(brand_subgroup)
    bg_array = []
    for i in range(0, no_of_subgroup):
        t = 0
        bg_array.append(t)
    for i in range(0, no_of_subgroup):
        if brand_subgroup[i] == selected_brand:
            bg_array[i] = 1
        else:
            bg_array[i] = 0

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
    list_X = (
    bg_array[0], bg_array[1], bg_array[2], bg_array[3], bg_array[4], bg_array[5], bg_array[6], bg_array[7], bg_array[8],
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
    list_not_predictor = (y_output_float, selected_cmdyy, selected_brand, selected_style, selected_remark)
    list_download = list_X + list_not_predictor
    global df_download
    df_download = pd.DataFrame(list_download)

    # Create a Table if not exists
    #     conn = sqlite3.connect('yy predict.db')
    #     c = conn.cursor()
    #     c.execute('''CREATE TABLE if not exists prediction
    #                 (subgroup_1 text NOT NULL,
    #                 subgroup_2 text NOT NULL,
    #                 subgroup_3 text NOT NULL,
    #                 subgroup_4 text NOT NULL,
    #                 subgroup_5 text NOT NULL,
    #                 subgroup_6 text NOT NULL,
    #                 subgroup_7 text NOT NULL,
    #                 subgroup_8 text NOT NULL,
    #                 subgroup_9 text NOT NULL,
    #                 subgroup_10 text NOT NULL,
    #                 subgroup_11 text NOT NULL,
    #                 subgroup_12 text NOT NULL,
    #                 subgroup_13 text NOT NULL,
    #                 solid Integer NOT NULL,
    #                 stripe Integer NOT NULL,
    #                 check Integer NOT NULL,
    #                 One_Way_Match Integer NOT NULL,
    #                 Two_Way_Match Integer NOT NULL,
    #                 Long_Sleeve Integer NOT NULL,
    #                 Plan_Cut_Qty Real NOT NULL,
    #                 Repeat_X Real NOT NULL,
    #                 Repeat_Y Real NOT NULL,
    #                 Average_Neck_Size Real NOT NULL,
    #                 Double_Cuff Integer NOT NULL,
    #                 Marker_Width Real NOT NULL,
    #                 Regular_Fit Integer NOT NULL,
    #                 Slim_Fit Integer NOT NULL,
    #                 Extra_Slim_Fit Integer NOT NULL,
    #                 Predicted_yy Real NOT NULL,
    #                 Actual_CMD_yy Real NOT NULL,
    #                 Brand text NOT NULL,
    #                 Style text NOT NULL,
    #                 Remark text NOT NULL);''')
    #     conn.commit()
    #     conn.close()

    # Insert predicted record into database
    #     conn = sqlite3.connect('‪yy predict.db')
    #     c = conn.cursor()
    #     c.execute(f"INSERT INTO prediction VALUES ({bg_array[0]},{bg_array[1]},{bg_array[2]},{bg_array[3]},{bg_array[4]},{bg_array[5]},{bg_array[6]},{bg_array[7]},{bg_array[8]},{bg_array[9]},{bg_array[10]},{bg_array[11]},{bg_array[12]},{solid_predictor}, {stripe_predictor}, {check_predictor},{owm_predictor}, {twm_predictor}, {ls_predictor},{pcq_predictor}, {rx_predictor}, {ry_predictor}, {ans_predictor},{dc_predictor}, {mw_predictor},{cf_predictor}, {sf_predictor}, {esf_predictor},{y_output_float},{selected_cmdyy},{selected_style}, {selected_remark})")
    #     conn.commit()
    #     conn.close()

    #     conn = sqlite3.connect('‪yy predict.db')
    #     c = conn.cursor()
    #     statement = '''SELECT * FROM prediction'''
    #     c.execute(statement)
    #     global db_test
    #     db_test = pd.read_sql_query(statement,conn)
    #     conn.commit()
    #     conn.close()

    # return 'Predicted yy is {}'.format(y_output)

    return '"{}" and brand is "{}""{}""{}""{}""{}""{}""{}""{}""{}""{}""{}""{}""{}"'.format(y_output, bg_array[0],
                                                                                           bg_array[1], bg_array[2],
                                                                                           bg_array[3], bg_array[4],
                                                                                           bg_array[5], bg_array[6],
                                                                                           bg_array[7], bg_array[8],
                                                                                           bg_array[9], bg_array[10],
                                                                                           bg_array[11], bg_array[12])


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True
)
# Download the user input and predicted yy
def func(n_clicks):
    df_header = [brand_subgroup[0], brand_subgroup[1], brand_subgroup[2], brand_subgroup[3], brand_subgroup[4],
                 brand_subgroup[5],
                 brand_subgroup[6], brand_subgroup[7], brand_subgroup[8], brand_subgroup[9], brand_subgroup[10],
                 brand_subgroup[11],
                 brand_subgroup[12], 'Solid', 'Stripe', 'Chk', 'One_Way_Match', 'Two_Way_Match', 'L/S', 'Plan_Cut_Qty',
                 'Repeat_X', 'Repeat_Y', 'Average_Neck_Size', 'Double_Cuff', 'Marker_Width', 'Regular_Fit',
                 'Slim_Fit', 'Extra_Slim_Fit', 'Predicted_yy', 'Actual_CMD_yy', 'Brand', 'Style', 'Remark_Text']

    sql_header = ['subgroup_1', 'subgroup_2', 'subgroup_3', 'subgroup_4', 'subgroup_5', 'subgroup_6',
                  'subgroup_7', 'subgroup_8', 'subgroup_9', 'subgroup_10', 'subgroup_11', 'subgroup_12',
                  'subgroup_13', 'solid', 'stripe', 'chk', 'One_Way_Match', 'Two_Way_Match', 'Long_Sleeve',
                  'Plan_Cut_Qty',
                  'Repeat_X', 'Repeat_Y', 'Average_Neck_Size', 'Double_Cuff', 'Marker_Width', 'Regular_Fit',
                  'Slim_Fit', 'Extra_Slim_Fit', 'Predicted_yy', 'Actual_CMD_yy', 'Brand', 'Style', 'Remark_Text']

    csv_download = pd.DataFrame(np.array(df_download).reshape(1, -1), columns=df_header)
    global sql_download
    sql_download = pd.DataFrame(np.array(df_download).reshape(1, -1), columns=sql_header)

    # Update to SQL table
    conn = sqlite3.connect('‪yy predict.db')
    c = conn.cursor()
    sql_download.to_sql('prediction', conn, if_exists='append', index=False)

    return dcc.send_data_frame(csv_download.to_csv, "mydf.csv")


if __name__ == '__main__':
    app.run_server(port=4050)
