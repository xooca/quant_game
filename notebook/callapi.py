
import requests


class read_alphavantage_api:
    def __init__(self):
        self.dummy = 'test'

    def create_url(self,url=None):
        if url is None:
            endpoint_details = {'function': 'TIME_SERIES_DAILY',
                                'symbol': 'SBIN.BSE',
                                'interval': '1min',
                                'outputsize': 'full',
                                'datatype': None,
                                'apikey': '90L05WAV2MO0OQ0V'}
            sub_url = ''
            for i,j in endpoint_details.items():
                if j != 'None':
                    sub_url += f"{i}={j}&"
            self.url = f"https://www.alphavantage.co/query?{sub_url}"
            self.url = self.url[:-1]
        else:
            self.url = url
        print(f"API URL is {self.url}")

    def call_api(self,url=None):
        self.create_url(url)
        r = requests.get(self.url)
        data = r.json()
        return data