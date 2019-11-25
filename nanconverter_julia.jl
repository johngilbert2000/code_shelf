using DataFrames, DataFramesMeta, CSV, Random, Statistics
import Missings

function convertNaN!(df::DataFrame,cols=nothing)
    "Converts missing values to median values in a dataframe (for specified or all columns), drops remaining missing values"
    if cols == nothing
        cols = names(df)
    end
    
    for col in cols
        if (typeof(df[1,col]) != String) & (typeof(df[1,col]) != Missing)
            x = median(skipmissing(df[:,col]))
            df[ismissing.(df[!,col]), col] = x
        end
    end
    
    # remove rows with missing values / remove missing type
    df = dropmissing(df, disallowmissing=true)
    return df
end

# df = convertNaN!(df,[]) # drops missing values
# df = convertNaN!(df)    # converts all missing values to median values