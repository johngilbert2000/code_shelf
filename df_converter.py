import pandas as pd
import numpy as np

def NaN_converter(df, converted_value='median',verbose=False, thresh=0.10):
    "Converts NaNs to either column median"
    # drops col if quantity of NaNs exceeds thresh
    desired_cols = df.columns
    
    # Test for NaN values
    cols_with_NaN = []
    for col in desired_cols:
        NaN_exists = False
        if len(df[np.isnan(df[col])]) > 0:
            NaN_exists = True
            cols_with_NaN += [col]
#     if ((NaN_exists == False) and verbose):
#         print("No NaN values in data")
    
    # Convert NaN values, if any
    dropped = []
    dropped_cts = []
    for col in cols_with_NaN:
        ct = 0
        if converted_value == 'median':
            to_val = df[col].median()
            nan_df = df[np.isnan(df[col])] # grab all NaN items into a dataframe
            # Use indices from NaN dataframe to change NaN values in original dataframe
            for i in nan_df.index.tolist():
                df.at[int(i),col] = to_val
                ct += 1
            if (ct > len(df)*thresh):
                df.drop(columns=col,inplace=True)
                dropped += [col]
                dropped_cts += [ct]
            elif verbose:
                print("{} NaNs were converted to {} in column {}".format(ct,to_val,col))
        else:
            raise ValueError("Valid inputs for converted_value are 'median'")
            
    if verbose:
        print('\n')
        for i in range(len(dropped)):
            print(f'{dropped[i]} was dropped with {dropped_cts[i]} NaNs')
    return df


def remove_negs(df):
    "Removes rows containing negative values"
    negs = []
    for col in df.columns:
        negs += list(df[df[col]<0].index)
    negs = list(set(negs))
    df.drop(negs,inplace=True)
    return df

def change_negs(df, to_val=0):
    "Changes negative values to to_val"
    for col in df.columns:
        neg = list(df[df[col]<0].index)
        df.loc[neg,col] = 0
    return df
    