import pandas as pd
import pickle as pkl 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow import keras
 
def get_bolwer_avg_run(bowlers,bowler_summary):
    bowler_list=bowlers.split(',')
    bowler_list=[x.strip(' ') for x in bowler_list]
    bowler_details=bowler_summary[bowler_summary.bowler.isin(bowler_list)]
    weighted_eco=(bowler_details.total_runs.sum()/(bowler_details.no_of_bowls.sum()/6))*6
    return weighted_eco

def get_batsmen_avg_run(batsmen,batsmen_summary):
    batsmen_list=batsmen.split(',')
    batsmen_list=[x.strip(' ') for x in batsmen_list]
    batsmen_details=batsmen_summary[batsmen_summary.striker.isin(batsmen_list)]
    weighted_strike_runs=(batsmen_details.runs_off_bat.sum()/batsmen_details.no_of_bowls.sum())*36
    return weighted_strike_runs
    

def predictRuns(file_name):

    input_sheet = pd.read_csv(file_name)
    input_sheet['no_of_batsmen'] = input_sheet['batsmen'].str.count(",") + 1
    input_sheet['bowler'] = input_sheet['bowlers'].str.count(",") + 1

    venue_map = pd.read_csv('venue_mapping.csv')

    input_sheet=pd.merge(input_sheet,venue_map,on='venue',how='left')

    batsmen_summary = pd.read_csv('batsmen_summary.csv')
    bowler_summary = pd.read_csv('bowler_summary.csv')

    input_sheet['weighted_eco'] = input_sheet['bowlers'].apply(lambda x:get_bolwer_avg_run(x,bowler_summary))
    
    input_sheet['weighted_strike_runs'] = input_sheet['batsmen'].apply(lambda x:get_batsmen_avg_run(x,batsmen_summary))
    

    hot_encoder_venue = pkl.load(open('hotencoder.sav', 'rb'))
    
    venue_cat=pd.DataFrame(hot_encoder_venue.transform(input_sheet[['new_venue_name']]).toarray(),
                columns=hot_encoder_venue.get_feature_names(['venue']))
    
    input_sheet = pd.concat([input_sheet, venue_cat], axis=1)

    input_sheet= input_sheet.drop([ 'batting_team', 'bowling_team','batsmen','bowlers','venue','new_venue_name'], axis=1)

    model = pkl.load(open('finalized_model.sav', 'rb'))
    #model = keras.models.load_model('finalized_model.h5')

    prediction = model.predict(input_sheet)

    total_runs_decode = pkl.load(open('scaler_totalruns.sav', 'rb'))

    result = np.round(total_runs_decode.inverse_transform(prediction)).astype(int)

    return result
    