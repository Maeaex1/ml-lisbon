
import numpy as np


#data_path = "C:/Users/Maximilian/maxAnalytics/LisbonHousing/data/"


def clean_dataset(df):
    #Drop useless Features + Rows that have no price
    df = df.drop(columns=['Concelho', 'Distrito', 'Ref', 'Preço de arrendamento', 'Finalidade', 'Preço de trespasse', 'Sotão', 'Quarto', 'Sala', 'Corredor', 'Cozinha', 'Estacionamento', 'W.C.', 'Área Terreno'])
    df = df.dropna(subset=['Preço de venda', 'Área bruta'])
    df = df[df['Preço de venda'] != 'Sob Consulta']
    df.columns = ['no_bathroom', 'condition', 'district', 'price', 'no_rooms',
                  'estate_type', 'area', 'altitude', 'energy_efficiency',
                  'longitude', 'area_total', 'area_usage']

    # Clean data from substrings and convert to numeric values
    df.area_total = df.area_total.str.replace("m2", "").str.replace(',', '.').astype(float)
    df.price = df.price.str.replace('€', '').str.replace('.', '').astype(float)
    df['energy_efficiency'] = df['energy_efficiency'].replace('[^a-zA-Z0-9 ]', '', regex=True)
    df[['area_usage', 'no_rooms', 'no_bathroom']] = df[['area_usage', 'no_rooms', 'no_bathroom']].astype(float)
    df = df.drop_duplicates()
    df = df.replace(r'^\s*$', np.nan, regex=True)

    return df

def create_features(df):
    df['sqm_price_usage'] = df['price'] / df['area_usage']
    df['sqm_price_total'] = df['price'] / df['area_total']
    df['avg_room_size'] = df['area_usage'] / df['no_rooms']

    return df


def impute_missing(data):
    data['energy_efficiency'] = data['energy_efficiency'].fillna("Unknown") # As maybe no test was conducted
    data['area'] = data['area'].fillna("None") # May comment line as too many districts mentioned
    data['district'] = data['district'].fillna("None")
    data['condition'] = data['condition'].fillna("None")

    data['area_usage'] = data['area_usage'].fillna(data['area_total'])

    cols = ['avg_room_size', 'no_rooms', 'no_bathroom', 'sqm_price_total', 'sqm_price_usage']

    for col in cols:
        data[col] = data[col].fillna(data[col].median())
        data[col] = data[col].replace(np.inf, data[col].median())
        #all_data[col] = all_data[col].astype(str)

    data['no_rooms'] = data['no_rooms'].replace(0, "None")

    return data

def remove_outliers(df_data):
    p_max_px = df_data['price'].quantile(0.99)
    p_min_px = df_data['price'].quantile(0.01)
    p_max_size = df_data['area_total'].quantile(0.99)

    clean_data = df_data[(df_data['price'] > p_min_px) & (df_data['price'] < p_max_px) & (df_data['area_total'] < p_max_size)]

    return clean_data

