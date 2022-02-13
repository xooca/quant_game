import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import numpy as np
from pathlib import Path
import os
from random import randrange
from datetime import timedelta  

class yahoo_finance:
    def __init__(self,ticker,save_path):
        self.ticker = ticker
        self.save_path = save_path + f'/{self.ticker}'
        self.stats_url = 'https://in.finance.yahoo.com/quote/'+ticker+'/key-statistics?p='+ticker
        self.hist_url = f'https://in.finance.yahoo.com/quote/{ticker}/'
        self.finance_url = f'https://in.finance.yahoo.com/quote/{ticker}/financials?p={ticker}'
        self.balance_sheet_url = f'https://in.finance.yahoo.com/quote/{ticker}/balance-sheet?p={ticker}'
        self.cash_flow_url = f'https://in.finance.yahoo.com/quote/{ticker}/cash-flow?p={ticker}'
        self.analysis_url = f'https://in.finance.yahoo.com/quote/{ticker}/analysis?p={ticker}'
        self.holder_url = f'https://in.finance.yahoo.com/quote/{ticker}/holders?p={ticker}'
        self.insider_roster_url = f'https://in.finance.yahoo.com/quote/{ticker}/insider-roster?p={ticker}'
        self.insider_trans_url = f'https://in.finance.yahoo.com/quote/{ticker}/insider-transactions?p={ticker}'
        self.summary_url = f'https://in.finance.yahoo.com/quote/{ticker}?p={ticker}'
        if not os.path.exists(Path(self.save_path)):
            os.makedirs(Path(self.save_path))

    def get_historical(self,frequency,start_date,end_date):
        start_date = np.int64(time.mktime(datetime.strptime(str(start_date), "%Y-%m-%d").timetuple()))
        end_date = np.int64(time.mktime(datetime.strptime(str(end_date), "%Y-%m-%d").timetuple()))
        #save_path_main = Path(self.save_path + f'/hist_{start_date}_{end_date}.csv')
        #save_path_other = Path(self.save_path + f'/hist_other_{start_date}_{end_date}.csv')
        self.hist_url_ = self.hist_url + f"history?period1={start_date}&period2={end_date}\
                &interval={frequency}&filter=history&frequency={frequency}"
        #print(self.hist_url_)
        page = requests.get(self.hist_url_)
        time.sleep(randrange(4,14))
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
        self.historical = pd.DataFrame(temp_dir).T.iloc[1:]
        self.historical.columns = pd.DataFrame(temp_dir).T.iloc[0].tolist()
        if len(temp_dir_div):
            self.historical_other = pd.DataFrame(temp_dir_div)
            self.historical_other = self.historical_other.T
        else:
            self.historical_other = pd.DataFrame()
        return self.historical,self.historical_other
        #self.historical.to_csv(save_path_main,index=True)
        #self.historical_other.to_csv(save_path_other,index=True)

    def get_historical_bydays(self,start_date,end_date,frequency):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date is None:
            end_date = datetime.now().date()
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        hist_list = []
        hist_other_list = []
        save_path_main = Path(self.save_path + f'/hist_{start_date}_{end_date}.csv')
        save_path_other = Path(self.save_path + f"/hist_other_{start_date}_{end_date}.csv")
        end_date_inc = start_date + timedelta(days=100)
        while True :
            hist,hist_other = self.get_historical(frequency,start_date,end_date_inc)
            hist_list.append(hist)
            if len(hist_other)>0:
                hist_other_list.append(hist_other)
            if end_date_inc > end_date:
                break
            start_date = start_date + timedelta(days=101)
            end_date_inc = end_date_inc + timedelta(days=100)
            #print(start_date,end_date_inc,end_date,len(hist_list),hist.shape,hist_other.shape)
        self.historical = pd.concat(hist_list)
        self.historical_other = pd.concat(hist_other_list)
        self.historical.to_csv(save_path_main,index=True)
        self.historical_other.to_csv(save_path_other,index=True)

    def get_statistics(self):
        page = requests.get(self.stats_url)
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
        self.statistics = pd.DataFrame(temp_dir).T
        self.statistics.columns = ['stats','value']
        save_path_main = Path(self.save_path + f'/stats_{datetime.now().year}.csv')
        self.statistics.to_csv(save_path_main,index=True)

    def get_analysis(self):
        page = requests.get(self.analysis_url)
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
        if len(temp_dir)>0:
            self.analysis = pd.DataFrame(temp_dir).T
            self.analysis.columns = ['value1','value2','value3','value4']
            save_path_main = Path(self.save_path + f'/analysis_{str(datetime.now().date())}.csv')
            self.analysis.to_csv(save_path_main,index=True)

    def get_holders(self):
        page = requests.get(self.holder_url)
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
        self.major_holder = pd.DataFrame(temp_dir_mh).T
        if len(self.major_holder):
            self.major_holder = self.major_holder.reset_index( drop=False)
            self.major_holder.columns = ['breakdown','description']
            save_path_main = Path(self.save_path + f'/major_holders_{str(datetime.now().date())}.csv')
            self.major_holder.to_csv(save_path_main,index=False)
        self.mf_holder = pd.DataFrame(temp_dir_mf).T.iloc[1:]
        if len(self.mf_holder):
            self.mf_holder.columns = pd.DataFrame(temp_dir_mf).T.iloc[0].tolist()
            save_path_main = Path(self.save_path + f'/mf_holder_{str(datetime.now().date())}.csv')
            self.mf_holder.to_csv(save_path_main,index=True)

    def get_insider_roster(self):
        page = requests.get(self.insider_roster_url)
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
        if len(temp_dir)>0:
            self.insider_holders = pd.DataFrame(temp_dir).T.iloc[1:]
            self.insider_holders.columns = pd.DataFrame(temp_dir).T.iloc[0].tolist()
            save_path_main = Path(self.save_path + f'/insider_roster_{str(datetime.now().date())}.csv')
            self.insider_holders.to_csv(save_path_main,index=True)

    def get_financial(self):
        page = requests.get(self.finance_url)
        table_tag='W(100%) Whs(nw) Ovx(a) BdT Bdtc($seperatorColor)'
        #list_len=6
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
        for t in tabl:
            rows = t.find_all("div")
            for row in rows:
                val = row.get_text(separator='|').split("|")
                if len(val)==list_len:
                    temp_dir[val[0]]=val[1:]
        if len(temp_dir)>0:
            self.financial = pd.DataFrame(temp_dir).T.iloc[1:]
            #self.financial.columns = pd.DataFrame(temp_dir).T.iloc[0].tolist()
            self.financial.columns = columns
            save_path_main = Path(self.save_path + f'/financial_{str(datetime.now().date())}.csv')
            self.financial.to_csv(save_path_main,index=True)

    def get_balancesheet(self):
        page = requests.get(self.balance_sheet_url)
        table_tag='W(100%) Whs(nw) Ovx(a) BdT Bdtc($seperatorColor)'
        #list_len=5
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
        for t in tabl:
            rows = t.find_all("div")
            for row in rows:
                val = row.get_text(separator='|').split("|")
                if len(val)==list_len:
                    temp_dir[val[0]]=val[1:]
        self.balance_sheet = pd.DataFrame(temp_dir).T.iloc[1:]
        #self.balance_sheet.columns = pd.DataFrame(temp_dir).T.iloc[0].tolist()
        self.balance_sheet.columns = columns
        save_path_main = Path(self.save_path + f'/balancesheet_{str(datetime.now().date())}.csv')
        self.balance_sheet.to_csv(save_path_main,index=True)

    def get_cashflow(self):
        page = requests.get(self.cash_flow_url)
        table_tag='W(100%) Whs(nw) Ovx(a) BdT Bdtc($seperatorColor)'
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
        for t in tabl:
            rows = t.find_all("div")
            for row in rows:
                val = row.get_text(separator='|').split("|")
                if len(val)==list_len:
                    temp_dir[val[0]]=val[1:]
        self.cash_flow = pd.DataFrame(temp_dir).T.iloc[1:]
        self.cash_flow.columns = columns
        save_path_main = Path(self.save_path + f'/cashflow_{str(datetime.now().date())}.csv')
        self.cash_flow.to_csv(save_path_main,index=True)

    def get_insider_transaction(self):
        page = requests.get(self.insider_trans_url)
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
                    #print(val)
                    temp_dir_intp[val[1]]=[val[2]]
                if len(val)==3:
                    temp_dir_insp[val[0]]=[val[1],val[2]]  
        if len(temp_dir)>0:
            self.insider_trans = pd.DataFrame(temp_dir).T.reset_index(drop=False)
            self.insider_trans.columns = [ 'Insider','Insider Type','Type', 'Date', 'Shares']
            save_path_main = Path(self.save_path + f'/insider_trans_{str(datetime.now().date())}.csv')
            self.insider_trans.to_csv(save_path_main,index=False)
        if len(temp_dir_insp)>0:
            self.insider_purch = pd.DataFrame(temp_dir_insp).T.iloc[1:]
            self.insider_purch.columns = pd.DataFrame(temp_dir_insp).T.iloc[0].tolist()
            self.insider_purch = self.insider_purch.iloc[:-1]
            save_path_main = Path(self.save_path + f'/insider_purch_{str(datetime.now().date())}.csv')
            self.insider_purch.to_csv(save_path_main,index=True)
        if len(temp_dir_intp)>0:
            self.institutional_purch = pd.DataFrame(temp_dir_intp).T.reset_index(drop=False)
            self.institutional_purch.columns = ['Net institutional purchases - Prior quarter to latest quarter', 'Shares']
            save_path_main = Path(self.save_path + f'/institutional_purch_{str(datetime.now().date())}.csv')
            self.institutional_purch.to_csv(save_path_main,index=False)

    def get_summary(self):
        page = requests.get(self.summary_url)
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
                #if val[0]=='Earnings date':
                #    temp_dir[val[0]]=[val[1]+val[2]+val[3]]
        self.summary = pd.DataFrame(temp_dir).T
        self.summary.columns =['data']
        save_path_main = Path(self.save_path + f'/summary_{str(datetime.now().date())}.csv')
        self.summary.to_csv(save_path_main,index=True)

    def run_stock_metadata(self):
        self.get_summary()
        self.get_insider_transaction()
        self.get_cashflow()
        self.get_balancesheet()
        time.sleep(randrange(1,4))
        self.get_financial()
        self.get_insider_roster()
        self.get_holders()
        self.get_analysis()
        self.get_statistics()
        #self.get_historical(hist_frequency,hist_start_date,hist_end_date)

    def run_historical(self,start_date,end_date,frequency='1d'):
        self.get_historical_bydays(start_date,end_date,frequency)
