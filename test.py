import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
                                        ### INTRODUCCION DATA SCIENCE MODULO 3 ###

                                        ### PREGUNTA 1 ####

# pd.set_option('display.width', 50000)

def energy():
    pd.set_option('display.max_columns', 5000)
    pd.set_option('display.max_rows', 5000)


    energy = pd.read_excel('Energy Indicators.xls')
    energy = energy.iloc[16:243]
    energy = energy[['Environmental Indicators: Energy', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5']].copy()
    energy = energy.rename(columns={'Environmental Indicators: Energy': 'Country', 'Unnamed: 3': 'Energy Supply',
                                    'Unnamed: 4': 'Energy Supply per Capita', 'Unnamed: 5': '% Renewable'})
    energy = energy.replace('...', np.nan)
    energy['Energy Supply'] = energy['Energy Supply'] * 1000000
    energy['Country'] = energy['Country'].str.split('\s\(').apply(lambda x: x[0])
    energy['Country'] = energy['Country'].str.replace('\d+', '')
    energy = energy.replace("Republic of Korea", "South Korea")
    energy = energy.replace("United States of America", "United States")
    energy = energy.replace("United Kingdom of Great Britain and Northern Ireland", "United Kingdom")
    energy = energy.replace("China, Hong Kong Special Administrative Region", "Hong Kong")
    energy = energy.reset_index()
    energy = energy[['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']]
    return energy


def GDP():
    GDP = pd.read_csv('world_bank.csv')
    s = (GDP.iloc[3].values)[:4].astype(str).tolist() + (GDP.iloc[3].values)[4:].astype(int).astype(str).tolist()
    GDP = GDP.iloc[4:]
    GDP.columns = s
    GDP = GDP[['Country Name', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    GDP.columns = ['Country', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    GDP = GDP.replace("Korea, Rep.", "South Korea", regex=False)
    GDP = GDP.replace("Iran, Islamic Rep.", "Iran")
    GDP = GDP.replace("Hong Kong SAR, China", "Hong Kong", regex=False)
    return GDP

def ScimEn():
    ScimEn = pd.read_excel('scimagojr-3.xlsx')
    # ScimEn = ScimEn.iloc[0:15]
    return ScimEn

def answer_one():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]

    return df

# print(answer_one())

                                        ##### PREGUNTA 2 #####



def answer_two():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')

    df1 = pd.merge(e, g, how='outer', left_on='Country', right_on='Country')
    df1 = pd.merge(s, df1, how='outer', left_on='Country', right_on='Country')
    df1 = df1.set_index('Country')

    return len(df1) - len(df)

# print(answer_two())

                                        ##### PREGUNTA 3 #####

def average(row):
    data = row[['2006',
                '2007',
                '2008',
                '2009',
                '2010',
                '2011',
                '2012',
                '2013',
                '2014',
                '2015']]
    avgGDPcon = pd.Series({'PIB': np.mean(data)})

    return avgGDPcon

def answer_three():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]
    df = df.apply(average, axis=1).sort_values(by=['PIB'], ascending=False)
    AvgGDPcon = df.iloc[:,0]

    return AvgGDPcon

# print(answer_three())

                                        ########## PREGUNTA 4 #########

def answer_four():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]
    df = (df.iloc[3, 19] - df.iloc[3, 10])

    return df

# print(answer_four())

                                        ########## PREGUNTA 5 #########

def answer_five():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]
    df = df['Energy Supply per Capita'].mean()

    return df

# print(answer_five())
                                        ########## PREGUNTA 6 #########



def answer_six():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]
    lista = []

    x = df['% Renewable'].max()
    y = df['% Renewable'].idxmax()
    lista.append(y)
    lista.append(x)
    t = tuple(lista)
    return t

                                        ########## PREGUNTA 7 #########

def answer_seven():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]
    lista = []
    df['new_columns'] = df['Self-citations'] / df['Citations']
    x = df['new_columns'].max()
    y = df['new_columns'].idxmax()
    lista.append(y)
    lista.append(x)
    t = tuple(lista)

    return t



                                        ########## PREGUNTA 8 #########

def answer_eight():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]
    df = df[['Energy Supply', 'Energy Supply per Capita']]
    df['Population'] = df['Energy Supply'] / df['Energy Supply per Capita']
    df = df.sort_values(by=['Population'], ascending=False)
    df = df.index[2]


    return df

# print(answer_eight())

                                        ########## PREGUNTA 9 #########


def nine():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]
    df['Population'] = df['Energy Supply'] / df['Energy Supply per Capita']
    df['Documents citables per capita'] = df['Citable documents'] / df['Population']
    df = df[['Documents citables per capita', 'Energy Supply per Capita']]
    y = df['Documents citables per capita'].corr(df['Energy Supply per Capita'], method= 'pearson')

    return y

# print(nine())

                                        ########## PREGUNTA 10 #########

def teen():
    e = energy()
    g = GDP()
    s = ScimEn()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df = df.iloc[0:15]
    y = df['% Renewable'].median()
    lista = []

    for i in range(len(df)):
        if df['% Renewable'][i] >= y:
            lista.append(1)


        elif df['% Renewable'][i] < y:
            lista.append(0)


    HighRenewcuyo = pd.Series(lista, index=[df.index])


    return HighRenewcuyo

# print(teen())


                                        ########## PREGUNTA 11 #########
def ContinentDict():
    ContinentDict = pd.DataFrame({'Continent':['Asia', 'North America', 'Asia', 'Europe', 'Europe', 'North America', 'Europe',\
                    'Asia', 'Europe', 'Asia', 'Europe', 'Europe', 'Asia', 'Australia','South America'],\
                    'Country':['China', 'United States','Japan', 'United Kingdom', 'Russian Federation', 'Canada',\
                    'Germany', 'India', 'France', 'South Korea', 'Italy', 'Spain', 'Iran', 'Australia', 'Brazil']})


    return ContinentDict



def eleven():
    e = energy()
    g = GDP()
    s = ScimEn()
    c = ContinentDict()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(df, c, how='inner', left_on='Country', right_on='Country')
    df = df.iloc[0:15]
    df['Population'] = df['Energy Supply'] / df['Energy Supply per Capita']
    df = df.set_index('Continent').groupby(level=0)\
        ['Population'].agg({'size': np.size, 'sum': np.sum, 'mean': np.average, 'std': np.nanstd})
    # df = df.groupby('Continent')['Population'].agg({'size': np.size, 'sum': np.sum, 'mean': np.average, 'std': np.nanstd})
    # df = df.groupby('Continent').agg({'Population':['size', 'sum', 'mean','std']})


    return df


# print(eleven())

                                        ########## PREGUNTA 12 #########


def twelve():
    e = energy()
    g = GDP()
    s = ScimEn()
    c = ContinentDict()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(df, c, how='inner', left_on='Country', right_on='Country')
    df = df.iloc[0:15]

    return df


                                        ########## PREGUNTA 13 #########

def estimate(row):
    data = row['Population']
    PopEst = pd.Series({'PostEst': '{:,}'.format(data)})
    return PopEst



def thirteen():
    e = energy()
    g = GDP()
    s = ScimEn()
    c = ContinentDict()
    df = pd.merge(e, g, how='inner', left_on='Country', right_on='Country')
    df = pd.merge(s, df, how='inner', left_on='Country', right_on='Country')
    df = df.set_index('Country')
    df['Population'] = df['Energy Supply'] / df['Energy Supply per Capita']
    df = df.iloc[0:15]
    df = df.apply(estimate, axis=1)
    PopEst = df.iloc[:, 0]

    return PopEst

# print(thirteen())

                                   ### INTRODUCCION DATA SCIENCE MODULO 4 ###
                                             ### PREGUNTA 1 ###


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)


def get_list_of_university_towns():
    university_towns = pd.read_table('university_towns.txt', sep='\n',header=None, names=['raw'], encoding='UTF-8')
    university_towns['State'] = university_towns['raw'].str.extract('^(.+)\[edit\]$', expand=True).fillna(method='ffill')
    university_towns['RegionName'] = university_towns['raw'].str.extract('(.+?)(?:\s\(|:).*', expand=True)
    university_towns = university_towns.drop('raw', axis=1)
    university_towns = university_towns.replace("The Colleges of Worcester Consortium", "The Colleges of Worcester Consortium:")
    university_towns = university_towns.replace("The Five College Region of Western Massachusetts", "The Five College Region of Western Massachusetts:")
    university_towns.at[240, 'RegionName'] = 'Faribault, South Central College'
    university_towns.at[246, 'RegionName'] = 'North Mankato, South Central College'
    university_towns = university_towns.dropna()
    university_towns.reset_index(inplace=True, drop=True)
    university_towns['State'].str.strip()
    university_towns['RegionName'].str.strip()
    return university_towns

# print(get_list_of_university_towns())

                                            ### PREGUNTA 2 ###

def get_recession_start():
    gdplev = pd.read_excel('gdplev.xls')
    gdplev = gdplev.iloc[7:287]
    gdplev = gdplev.drop(['Current-Dollar and "Real" Gross Domestic Product', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3' \
                             ,'Unnamed: 5'], axis=1)
    gdplev = gdplev.iloc[212:287,0:2].reset_index(drop=True)
    gdplev = gdplev.rename(columns={'Unnamed: 4': 'GDP', 'Unnamed: 6':'GDPGDP'})


    for i in range(0,len(gdplev)):
        if (gdplev['GDPGDP'][i] > gdplev['GDPGDP'][i+1] and gdplev['GDPGDP'][i+1] > gdplev['GDPGDP'][i+2] \
                and gdplev['GDPGDP'][i+2] > gdplev['GDPGDP'][i+3]):
            if (gdplev['GDPGDP'][i+3] < gdplev['GDPGDP'][i+4] and gdplev['GDPGDP'][i+4] < gdplev['GDPGDP'][i+5]):

                y = gdplev['GDP'][i]
                break
    return y

# print(get_recession_start())
                                            ### PREGUNTA 3 ###


def get_recession_end():
    gdplev = pd.read_excel('gdplev.xls')
    gdplev = gdplev.iloc[7:287]
    gdplev = gdplev.drop(['Current-Dollar and "Real" Gross Domestic Product', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3' \
                             ,'Unnamed: 5'], axis=1)
    gdplev = gdplev.iloc[212:287,0:2].reset_index(drop=True)
    gdplev = gdplev.rename(columns={'Unnamed: 4': 'GDP', 'Unnamed: 6':'GDPGDP'})

    for i in range(0,len(gdplev)):
        if (gdplev['GDPGDP'][i] > gdplev['GDPGDP'][i+1] and gdplev['GDPGDP'][i+1] > gdplev['GDPGDP'][i+2] \
                and gdplev['GDPGDP'][i+2] > gdplev['GDPGDP'][i+3]):
            if (gdplev['GDPGDP'][i+3] < gdplev['GDPGDP'][i+4] and gdplev['GDPGDP'][i+4] < gdplev['GDPGDP'][i+5]):

                y =gdplev['GDP'][i+5]
                break
    return y

# print(get_recession_end())
                                            ### PREGUNTA 4 ####


def get_recession_bottom():
    gdplev = pd.read_excel('gdplev.xls')
    gdplev = gdplev.iloc[7:287]
    gdplev = gdplev.drop(['Current-Dollar and "Real" Gross Domestic Product', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3' \
                             ,'Unnamed: 5'], axis=1)
    gdplev = gdplev.iloc[212:287,0:2].reset_index(drop=True)
    gdplev = gdplev.rename(columns={'Unnamed: 4': 'GDP', 'Unnamed: 6':'GDPGDP'})

    for i in range(0,len(gdplev)):
        if (gdplev['GDPGDP'][i] > gdplev['GDPGDP'][i+1] and gdplev['GDPGDP'][i+1] > gdplev['GDPGDP'][i+2] \
                and gdplev['GDPGDP'][i+2] > gdplev['GDPGDP'][i+3]):
            if (gdplev['GDPGDP'][i+3] < gdplev['GDPGDP'][i+4] and gdplev['GDPGDP'][i+4] < gdplev['GDPGDP'][i+5]):
                y = (gdplev['GDP'][i+2])
                break
    return y

# print(get_recession_bottom())


                                            ### PREGUNTA 5 ###
def convert_housing_data_to_quarters():

    City_Zhvi_AllHomes = pd.read_csv('City_Zhvi_AllHomes.csv', encoding='latin-1')

    states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National',\
              'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana',\
              'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho',\
              'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan',\
              'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico',\
              'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa',\
              'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana',\
              'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California',\
              'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island',\
              'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia',\
              'ND': 'North Dakota', 'VA': 'Virginia'}
    City_Zhvi_AllHomes['State'] = City_Zhvi_AllHomes['State'].map(states)
    City_Zhvi_AllHomes = City_Zhvi_AllHomes.set_index(['State', 'RegionName'])
    City_Zhvi_AllHomes = City_Zhvi_AllHomes.iloc[:, 49:249]
    City_Zhvi_AllHomes.columns = pd.to_datetime(City_Zhvi_AllHomes.columns)
    City_Zhvi_AllHomes = City_Zhvi_AllHomes.resample('Q', axis=1).mean()
    City_Zhvi_AllHomes = City_Zhvi_AllHomes.rename(columns=lambda x: str(x.to_period('Q')).lower())


    return City_Zhvi_AllHomes

# print(convert_housing_data_to_quarters())

# City_Zhvi_AllHomes = pd.read_csv('City_Zhvi_AllHomes.csv', encoding='latin-1')

                                                    ### PREGUNTA 6###


def run_ttest():
    hdf = convert_housing_data_to_quarters()
    rec_start = get_recession_start()
    rec_bottom = get_recession_bottom()
    ul = get_list_of_university_towns()

    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''

    from scipy.stats import ttest_ind

    hdf = convert_housing_data_to_quarters()
    rec_start = get_recession_start()
    rec_bottom = get_recession_bottom()
    ul = get_list_of_university_towns()

    quarter_before_recession = hdf['2008q2']
    hdf['Precio_Ratio'] = quarter_before_recession.divide(hdf[rec_bottom])

    ul_list = ul.to_records(index=False).tolist()
    group1 = hdf.loc[ul_list]
    group2 = hdf.loc[~hdf.index.isin(ul_list)]
    df = pd.merge(hdf.reset_index(), ul, on=ul.columns.tolist(), indicator='_flag', how='outer')
    unitowns = df[df['_flag'] == 'both']
    nonunitowns = df[df['_flag'] != 'both']

    t = ttest_ind(unitowns['Precio_Ratio'], nonunitowns['Precio_Ratio'], nan_policy='omit')

    u = t[1]
    p = 0.01

    y = unitowns['Precio_Ratio'].mean()
    z = nonunitowns['Precio_Ratio'].mean()

    if y < z:
        n = 'university town'

    elif y > z:
        n = 'non-university town'

    if u < p:
        v = True

    elif u > p:
        v = False

    result = (v, u, n)


    return result


class Coche():

    def __init__(self):

        self.largochasis = 250
        self.anchochasis = 120
        self.__ruedas = 4
        self.enmarcha = False

    def arrancar(self):
        self.enmarcha = True

    def estado(self):
        if self.enmarcha is True:
            return 'El coche esta en marcha'

        else:

            return 'El coche esta parado'

# micoche=Coche()                                                    # instanciar una clase #



# print(micoche.estado())
#
# print('----------------ACONTINUACION CREAMOS EL SEGUNDO OBJETO--------------------')
#
# micoche2 = Coche()
# print(micoche2.largochasis)
# print(micoche2.__ruedas)
# print(micoche2.estado())

# mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
# {'a': 100, 'b': 200, 'c': 300, 'd': 400},
# {'a': 20000, 'b': 200, 'c': 300, 'd': 400},
# {'a': 100, 'b': 200, 'c': 300, 'd': 400},
# {'a': 100, 'b': 200, 'c': 300, 'd': 400},
# {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
#
# df = pd.DataFrame(mydict)
# print(df)
# print('----------------------------')

import re

hash = '#python#wisdom#programming#hack'

print(re.split(r'#', hash))