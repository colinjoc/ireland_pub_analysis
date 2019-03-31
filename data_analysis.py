
import requests
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp


def get_pages():
    """
    connect to the guindex website and download the pub data from their API. Decode and dump the json into a file.
    """
    all_data=[]
    for x in range(1, 40): # 39 pages of results are currently on the website.
        response = requests.get("https://guindex.ie/api/pubs/?page=%s"%x)    
        res_j = json.loads(response.content.decode('utf-8'))           
        all_data+=res_j["results"]
        print (len(all_data))
    json.dump(res_j,open("save_all_pubs.json","a"))

def get_small_data():
    """
    just returns the first 100 pubs in the database
    """
    response = requests.get("https://guindex.ie/api/pubs/")
    res_j = json.loads(response.content.decode('utf-8'))        
    json.dump(res_j,open("small_pubs.json","a"))
    return True

def load_pub():
    """
    load up the json file containing pubs
    """
    import json
    data=json.load(open("all_pubs.json","r"))
    return data


data = load_pub()
df = pd.DataFrame(data)
print("Total number of pubs in the database:",len(df))

#remove pubs that are closed from the analysis
df_open=df[df["closed"]==False]
print("Total number of open pubs in the database:",len(df_open))

# number of pubs with a logged price
#first convert prices to numbers
df["lastPrice"]=pd.to_numeric(df['lastPrice'])
dfp=df.copy()
dfp=dfp.dropna(subset=['lastPrice'])
print("Total number of pubs with a submitted price:",len(dfp))

cols=["latitude","longitude"]
df[cols]=df[cols].apply(pd.to_numeric)
dfp[cols]=df[cols].apply(pd.to_numeric)


def plot_ireland():
    """
    if we plot latitude vs longitude, can we recognise the outline of ireland?

    putting into a function to keep things tidy
    """
    cols=["latitude","longitude"]
    df[cols]=df[cols].apply(pd.to_numeric)
    dfp[cols]=df[cols].apply(pd.to_numeric)
    df.plot.scatter('longitude','latitude',c="green")
    dfp.plot.scatter('longitude','latitude',c="black")
    plt.show()

def long_lat_price():
    """
    looking for patterns in price vs longitude and price vs lattitude
    """
    g = sns.PairGrid(dfp[['longitude', 'latitude', 'lastPrice']])    
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    plt.show()

print price_distribution():
    # can see a difference in price, are they normally distributed?
    print( dfp["lastPrice"].describe())
    dfp['lastPrice'].hist(bins = np.arange(3.7,8,0.1))
    plt.show()
    # can see its a bit skewed. Can check if distribution is normally distributed using the 

    shapiro_test=sp.stats.shapiro(dfp["lastPrice"])
    norm_test = sp.stats.normaltest(dfp["lastPrice"])
    print("shapiro p value: %s, chi-squared test: %s."%(shapiro_test[1],norm_test[1])) 
    # both are less than 0.05, data is not a normal distribution!


def price_by_county():
    # average price by county
    county_list = set(dfp["county"])
    for county_name in county_list:
        
        print( "Number of visity pubs in count %s: %s, Average price: %s" %(county_name, len(dfp[dfp["county"]==county_name]), round(dfp[dfp["county"]==county_name]["lastPrice"].mean(),2) ))
        print( dfp[dfp["county"]==county_name]['lastPrice'].describe())
        if len(dfp[dfp["county"]==county_name])>8:
            norm_test = sp.stats.normaltest(dfp[dfp["county"]==county_name]['lastPrice'])
            print("chi-squared test: %s."%(norm_test[1]),"\n") 
        else:
            print("Sample not large enough for chi-squared\n")


"""
Can we get any information from the names of pubs?
"""
# some names have a place-name in brackets at the end of the name. This was done in the database I believe to differentiate different pubs with the same name. I will remove them.
dfname=df.copy()
for x in df.index:
    name = dfname["name"][x]
    if "(" in dfname["name"][x]:
        dfname["name"][x]= name[:name.index("(")-1]

# get a list of common names!
print(dfname["name"].value_counts())
# You can see that some names are very frequent, i.e. the village inn!

# what is the length of each name?
dfname["nameLength"]=dfname.apply( lambda row: len(row["name"]), axis=1)


