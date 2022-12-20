
from kiteconnect import KiteConnect,KiteTicker
from selenium import webdriver
import time
import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from omegaconf import OmegaConf
from data.data_utils import initialize_config,print_log
import requests
from data.base.base_class import TradeApiData
import pandas as pd
from pyotp import TOTP
import datetime as dt
import duckdb


class read_alphavantage_api(TradeApiData):
    def __init__(self,master_config_path):
        TradeApiData.__init__(self,master_config_path)

    def create_url(self,url=None):
        if url is None:
            endpoint_details = dict(self.endpoint_details)
            sub_url = ''
            for i,j in endpoint_details.items():
                if j != 'None':
                    sub_url += f"{i}={j}&"
            self.url = f"{self.base_url}{sub_url}"
            self.url = self.url[:-1]
        else:
            self.url = url
        print_log(f"API URL is {self.url}",self.using_print)

    def call_api(self,url=None,read_action=None):
        self.create_url(url)
        if read_action is None:
            if self.read_action == 'json':
                r = requests.get(self.url)
                data = r.json()
            elif self.read_action == 'csv':
                data = pd.read_csv(self.url)
            else:
                print_log(f"Invalid read action: {self.read_action}",self.using_print)
                data = None
        return data
class trade_zerodha(TradeApiData):
    def __init__(self,master_config_path):
        TradeApiData.__init__(self,master_config_path)
        self.driver = self.create_chrome_browser()
        self.con = duckdb.connect(database=self.zerodha_duck_db_path, read_only=False)

    def create_chrome_browser(self):
        service = webdriver.chrome.service.Service('./chromedriver')
        service.start()
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options = options.to_capabilities()
        driver = webdriver.Remote(service.service_url, options)
        return driver

    def get_request_token(self):
        self.username_input = self.driver.find_element_by_xpath(self.zerodha_userid_xpath)
        self.password_input = self.driver.find_element_by_xpath(self.zerodha_password_xpath)
        self.username_input.send_keys(self.zerodha_userid)
        self.password_input.send_keys(self.zerodha_password)
        self.driver.find_element_by_xpath(self.zerodha_submit_pre_button).click()
        self.post_submit_button = self.driver.find_element_by_xpath(self.zerodha_submit_post_button)
        pin = self.driver.find_element_by_xpath(self.zerodha_pin_xpath)
        if self.zerodha_pin_flag is False:
            totp = TOTP(self.zerodha_topt)
            self.zerodha_pin = totp.now()
        pin.send_keys(self.zerodha_pin)
        self.driver.find_element_by_xpath(self.zerodha_submit_post_button).click()
        time.sleep(10)
        request_token=self.driver.current_url.split('=')[1].split('&action')[0]
        with open(self.zerodha_request_token_path, 'w') as the_file:
            the_file.write(request_token)
        self.driver.quit()

    def get_access_token(self):
        self.request_token = open(self.zerodha_request_token_path,'r').read()
        self.kite = KiteConnect(api_key=self.zerodha_api_key)
        self.session_data = self.kite.generate_session(self.request_token, api_secret=self.zerodha_api_secret)
        with open(self.zerodha_access_token_path, 'w') as file:
                file.write(self.session_data["access_token"])

    def create_session_with_access_tokenn(self):
        self.kite = KiteConnect(api_key=self.zerodha_api_key)
        access_token = open(self.zerodha_access_token_path,'r').read()
        self.kite.set_access_token(access_token)

    def get_instruments(self):
        instrument_dump = self.kite.instruments(self.zerodha_exchange)
        self.instrument_df = pd.DataFrame(instrument_dump)


    def token_lookup(self,symbol_list):
        token_list = []
        for symbol in symbol_list:
            token_list.append(int(self.instrument_df[self.instrument_df.tradingsymbol==symbol].instrument_token.values[0]))
        return token_list

    def place_market_order(self,symbol,buy_sell,quantity):    
        # Place an intraday market order on NSE
        if buy_sell == "buy":
            t_type=self.kite.TRANSACTION_TYPE_BUY
        elif buy_sell == "sell":
            t_type=self.kite.TRANSACTION_TYPE_SELL
        self.kite.place_order(tradingsymbol=symbol,
                        exchange=self.kite.EXCHANGE_NSE,
                        transaction_type=t_type,
                        quantity=quantity,
                        order_type=self.kite.ORDER_TYPE_MARKET,
                        product=self.kite.PRODUCT_MIS,
                        variety=self.kite.VARIETY_REGULAR)
    
    def cancel_order(self,order_id):    
        # Modify order given order id
        self.kite.cancel_order(order_id=order_id,
                        variety=self.kite.VARIETY_REGULAR)

    def place_bracket_order(self,symbol,buy_sell,quantity,atr,price):    
        # Place an intraday market order on NSE
        if buy_sell == "buy":
            t_type=self.kite.TRANSACTION_TYPE_BUY
        elif buy_sell == "sell":
            t_type=self.kite.TRANSACTION_TYPE_SELL
        self.kite.place_order(tradingsymbol=symbol,
                        exchange=self.kite.EXCHANGE_NSE,
                        transaction_type=t_type,
                        quantity=quantity,
                        order_type=self.kite.ORDER_TYPE_LIMIT,
                        price=price, #BO has to be a limit order, set a low price threshold
                        product=self.kite.PRODUCT_MIS,
                        variety=self.kite.VARIETY_BO,
                        squareoff=int(6*atr), 
                        stoploss=int(3*atr), 
                        trailing_stoploss=2)

    def on_ticks(self,ws,ticks):
        # Callback to receive ticks.
        #logging.debug("Ticks: {}".format(ticks))
        self.insert_ticker_table(self.)
        print(ticks)

    def on_connect(self,ws,response):
        # Callback on successful connect.
        # Subscribe to a list of instrument_tokens (RELIANCE and ACC here).
        #logging.debug("on connect: {}".format(response))
        ws.subscribe(self.tokens)
        ws.set_mode(ws.MODE_FULL,self.zerodha_tickers) # Set all token tick in `full` mode.
        #ws.set_mode(ws.MODE_FULL,[tokens[0]])  # Set one token tick in `full` mode.

    def stream_data(self):
        #access_token = open(self.zerodha_access_token_path,'r').read()
        self.kws = KiteTicker(self.zerodha_api_key,self.kite.access_token)
        self.get_instruments()
        self.tokens = self.token_lookup(self.zerodha_tickers)
        self.kws.on_ticks=self.on_ticks
        self.kws.on_connect=self.on_connect
        self.kws.connect()

    def fetch_ohlc_extended(self,ticker,inception_date, interval):
        """extracts historical data and outputs in the form of dataframe
        inception date string format - dd-mm-yyyy"""
        instrument = self.token_lookup([ticker])
        from_date = dt.datetime.strptime(inception_date, '%d-%m-%Y')
        to_date = dt.date.today()
        data = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        while True:
            if from_date.date() >= (dt.date.today() - dt.timedelta(100)):
                data = data.append(pd.DataFrame(self.kite.historical_data(instrument,from_date,dt.date.today(),interval)),ignore_index=True)
                break
            else:
                to_date = from_date + dt.timedelta(100)
                data = data.append(pd.DataFrame(self.kite.historical_data(instrument,from_date,to_date,interval)),ignore_index=True)
                from_date = to_date
        data.set_index("date",inplace=True)
        return data

    def square_off_all(self):
        #fetching orders and position information   
        a,b = 0,0
        while a < 10:
            try:
                pos_df = pd.DataFrame(self.kite.positions()["day"])
                break
            except:
                print("can't extract position data..retrying")
                a+=1
        while b < 10:
            try:
                ord_df = pd.DataFrame(self.kite.orders())
                break
            except:
                print("can't extract order data..retrying")
                b+=1

        #closing all open position      
        for i in range(len(pos_df)):
            ticker = pos_df["tradingsymbol"].values[i]
            if pos_df["quantity"].values[i] >0:
                quantity = pos_df["quantity"].values[i]
                self.place_market_order(ticker,"sell", quantity)
            if pos_df["quantity"].values[i] <0:
                quantity = abs(pos_df["quantity"].values[i])
                self.place_market_order(ticker,"buy", quantity)

        #closing all pending orders
        pending = ord_df[ord_df['status'].isin(["TRIGGER PENDING","OPEN"])]["order_id"].tolist()
        drop = []
        attempt = 0
        while len(pending)>0 and attempt<5:
            pending = [j for j in pending if j not in drop]
            for order in pending:
                try:
                    self.cancel_order(order)
                    drop.append(order)
                except:
                    print("unable to delete order id : ",order)
                    attempt+=1

    def create_ticker_table(self,ticker_name):
        self.con.execute(f'''
        CREATE TABLE main.{ticker_name}_ticker(
        instrument_token VARCHAR, 
        volume DECIMAL(8,2), 
        last_price DECIMAL(8,2),
        average_price DECIMAL(8,2),
        last_quantity INTEGER,
        buy_quantity INTEGER,
        sell_quantity INTEGER,
        last_trade_time DATE
        )
        ''')

    def insert_ticker_table(self,ticker_name,data_value):
        self.con.execute(f'''
        INSERT into main.{ticker_name}_ticker VALUES
        {data_value}
        ''')

    def create_ohlc_table(self,ticker_name):
        self.con.execute(f'''
        CREATE TABLE main.{ticker_name}_ohlc(
        trade_time DATE,
        instrument_token VARCHAR, 
        open DECIMAL(8,2), 
        high DECIMAL(8,2),
        low DECIMAL(8,2),
        close DECIMAL(8,2),
        volumn INTEGER
        )
        ''')

    def insert_ohlc_table(self,ticker_name,data_value):
        self.con.execute(f'''
        INSERT into main.{ticker_name}_ohlc VALUES
        {data_value}
        ''')

    def create_prediction_table(self,ticker_name):
        self.con.execute(f'''
        CREATE TABLE main.{ticker_name}_prediction(
        trade_time DATE,
        instrument_token VARCHAR, 
        label_name VARCHAR,
        prediction VARCHAR
        )
        ''')

    def insert_prediction_table(self,ticker_name,data_value):
        self.con.execute(f'''
        INSERT into main.{ticker_name}_prediction VALUES
        {data_value}
        ''')