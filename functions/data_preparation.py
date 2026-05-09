import pandas as pd
from tqdm import tqdm


def create_encoded_categories(data, columns_to_encode):
    "prepare table encoded_categories with labels for target/mean encoding of categorical columns"

    encoded_categories = pd.DataFrame()

    for column in tqdm(columns_to_encode):
        df = pd.DataFrame(data.groupby(column).klik.mean())
        df.columns = [f'{column}_encoded']
        df = df.reset_index()

        encoded_categories = pd.concat((encoded_categories, df), axis = 1)

    encoded_categories.to_csv('data/encoded_categories.csv')
    


def mean_encoding(data, columns_to_encode, encoded_categories):
    "target/mean encoding of categorical columns"
    
    for column in tqdm(columns_to_encode):
        data = data.set_index(column).join(encoded_categories[[column, f'{column}_encoded']].set_index(column))
        data.reset_index(drop = True, inplace = True)
        
        null_indexes =  data[data[f'{column}_encoded'].isnull()].index
        data.loc[null_indexes, f'{column}_encoded'] = 0
        
    return data