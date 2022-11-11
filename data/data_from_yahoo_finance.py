import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import datetime
import numpy as np

def load_hist_data_table_yf(url,table_tag):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table")
    temp_dir = {}
    temp_dir_div = {}
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            val = row.get_text(separator='|').split("|")
            if len(val)==7:
                temp_dir[val[0]]=val[1:]
            elif len(val)==4:
                temp_dir_div[val[0]]=val[1:]
    df = pd.DataFrame(temp_dir).T.iloc[1:]
    df.columns = pd.DataFrame(temp_dir).T.iloc[0].tolist()
    if len(temp_dir_div):
        df_other = pd.DataFrame(temp_dir_div)
        df_other = df_other.T
    else:
        df_other = pd.DataFrame()
    return df,df_other

def load_stats_data_table_yf(url,table_tag):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table")
    temp_dir = {}
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            val = row.get_text(separator='|').split("|")
            if len(val)>2:
                temp_dir[val[0]]=[val[1],val[-1]]
    df = pd.DataFrame(temp_dir).T
    df.columns = ['stats','value']
    return df

def load_analysis_data_table_yf(url):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table")
    temp_dir = {}
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            val = row.get_text(separator='|').split("|")
            if val[0]in['Revenue estimate','Earnings estimate','EPS trend','EPS revisions']:
                temp_dir[val[0]]=[val[1]+val[3]+val[4],val[5]+val[7]+val[8],val[9]+val[10],val[11]+val[12]]
            if len(val)==5:
                temp_dir[val[0]]=[val[1],val[2],val[3],val[4]]
    df = pd.DataFrame(temp_dir).T
    df.columns = ['value1','value2','value3','value4']
    return df

def load_holders_data_table_yf(url):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table")
    temp_dir_mh = {}
    temp_dir_mf = {}
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            val = row.get_text(separator='|').split("|")
            if len(val)==2:
                temp_dir_mh[val[0]]=[val[1]]
            if len(val)==5:
                temp_dir_mf[val[0]]=[val[1],val[2],val[3],val[4]]
    df_mh = pd.DataFrame(temp_dir_mh).T
    df_mh = df_mh.reset_index( drop=False)
    df_mh.columns = ['breakdown','description']
    df_mf = pd.DataFrame(temp_dir_mf).T.iloc[1:]
    df_mf.columns = pd.DataFrame(temp_dir_mf).T.iloc[0].tolist()
    return df_mh,df_mf

def load_insider_roster_data_table_yf(url):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table")
    temp_dir = {}
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            val = row.get_text(separator='|').split("|")
            if len(val)==5:
                temp_dir[val[0]]=[val[2],val[3],val[4]]
            if len(val)==4:
                temp_dir[val[0]]=[val[1],val[2],val[3]]
    df = pd.DataFrame(temp_dir).T.iloc[1:]
    df.columns = pd.DataFrame(temp_dir).T.iloc[0].tolist()
    return df

def load_financial_data_table_yf(url,table_tag,list_len):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.find_all('div', {'class': [table_tag]})
    temp_dir = {}
    for t in tabl:
        rows = t.find_all("div")
        for row in rows:
            val = row.get_text(separator='|').split("|")
            print(val,len(val))
            if len(val)==list_len:
                temp_dir[val[0]]=val[1:]
    df = pd.DataFrame(temp_dir).T.iloc[1:]
    print(df)
    df.columns = pd.DataFrame(temp_dir).T.iloc[0].tolist()
    return df

def load_cash_flow_data_table_yf(url,table_tag):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.find_all('div', {'class': [table_tag]})
    temp_dir = {}
    length = soup.find_all('div',{'class':{'D(tbr) C($primaryColor)'}})
    columns = []
    for t in length:
        rows = t.find_all("div")
        for row in rows:
            columns.append(row.get_text())
    list_len = len(columns[2:]) +1
    columns = columns[2:]
    #print(columns)
    #print(list_len)
    for t in tabl:
        rows = t.find_all("div")
        for row in rows:
            val = row.get_text(separator='|').split("|")
            #print(val,len(val))
            if len(val)==list_len:
                #print(val,len(val))
                temp_dir[val[0]]=val[1:]
    df = pd.DataFrame(temp_dir).T.iloc[1:]
    df.columns = columns
    return df

def load_insider_transactions_data_table_yf(url):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table")
    temp_dir = {}
    temp_dir_insp = {}
    temp_dir_intp = {}
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            val = row.get_text(separator='|').split("|")
            if len(val)==5:
                temp_dir[val[0]]=val[1:]
            if val[1] in ['Net shares purchased (sold)','% change in institutional shares held']:
                print(val)
                temp_dir_intp[val[1]]=[val[2]]
            if len(val)==3:
                temp_dir_insp[val[0]]=[val[1],val[2]]
    df = pd.DataFrame(temp_dir).T.reset_index(drop=False)
    df.columns = [ 'Insider','Insider Type','Type', 'Date', 'Shares']
    df_insp = pd.DataFrame(temp_dir_insp).T.iloc[1:]
    df_insp.columns = pd.DataFrame(temp_dir_insp).T.iloc[0].tolist()

    df_intp = pd.DataFrame(temp_dir_intp).T.reset_index(drop=False)
    df_intp.columns = ['Net institutional purchases - Prior quarter to latest quarter', 'Shares']
    return df,df_insp.iloc[:-1],df_intp

def load_summary_data_table_yf(url):
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table")
    temp_dir = {}
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            #print(row.get_text(separator='|'))
            val = row.get_text(separator='|').split("|")
            #print(val)
            if len(val)==2:
                temp_dir[val[0]]=[val[1]]
            if val[0]=='Earnings date':
                temp_dir[val[0]]=[val[1]+val[2]+val[3]]
    df = pd.DataFrame(temp_dir).T
    df.columns =['data']
    return df

def get_stats(ticker):
    url = 'https://in.finance.yahoo.com/quote/'+ticker+'/key-statistics?p='+ticker
    df = load_stats_data_table_yf(url,table_tag='Mstart(a) Mend(a)')
    return df

def get_hist_data(ticker,start_date,end_date,frequency):
    start_date = np.int64(time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
    end_date = np.int64(time.mktime(datetime.datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
    url = f'https://in.finance.yahoo.com/quote/{ticker}/history?period1={start_date}&period2={end_date}\
        &interval={frequency}&filter=history&frequency={frequency}'
    print(url)
    df = load_hist_data_table_yf(url,table_tag='Pb(10px) Ovx(a) W(100%)')
    return df

def get_financial_data(ticker):
    url = f'https://in.finance.yahoo.com/quote/{ticker}/financials?p={ticker}'
    print(url)
    df = load_financial_data_table_yf(url,table_tag='W(100%) Whs(nw) Ovx(a) BdT Bdtc($seperatorColor)',list_len=6)
    return df

def get_balancesheet_data(ticker):
    url = f'https://in.finance.yahoo.com/quote/{ticker}/balance-sheet?p={ticker}'
    print(url)
    df = load_financial_data_table_yf(url,table_tag='W(100%) Whs(nw) Ovx(a) BdT Bdtc($seperatorColor)',list_len=5)
    return df

def get_cashflow_data(ticker):
    url = f'https://in.finance.yahoo.com/quote/{ticker}/cash-flow?p={ticker}'
    print(url)
    df = load_cash_flow_data_table_yf(url,table_tag='W(100%) Whs(nw) Ovx(a) BdT Bdtc($seperatorColor)')
    #df = load_cash_flow_data_table_yf(url)
    return df

def get_analysis_data(ticker):
    url = f'https://in.finance.yahoo.com/quote/{ticker}/analysis?p={ticker}'
    print(url)
    df = load_analysis_data_table_yf(url)
    return df

def get_holders_data(ticker):
    url = f'https://in.finance.yahoo.com/quote/{ticker}/holders?p={ticker}'
    print(url)
    df1,df2 = load_holders_data_table_yf(url)
    return df1,df2

def get_inside_roster_data(ticker):
    url = f'https://in.finance.yahoo.com/quote/{ticker}/insider-roster?p={ticker}'
    print(url)
    df= load_insider_roster_data_table_yf(url)
    return df

def get_insider_transactions_data(ticker):
    url = f'https://in.finance.yahoo.com/quote/{ticker}/insider-transactions?p={ticker}'
    print(url)
    df= load_insider_transactions_data_table_yf(url)
    return df

def get_summary_data(ticker):
    url = f'https://in.finance.yahoo.com/quote/{ticker}?p={ticker}'
    print(url)
    df= load_summary_data_table_yf(url)
    return df