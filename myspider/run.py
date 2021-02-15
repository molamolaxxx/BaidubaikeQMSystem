from urllib import request
import requests
import numpy as np
import myspider.spider as sp
from myspider.saver import Saver
from config import SpiderConfig
class Runner():
    def __init__(self):
        self.headers = {
                'User-Agent': ' Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
                              ' AppleWebKit/537.36 (KHTML, like Gecko)'
                              ' Chrome/64.0.3282.186 Safari/537.36',
                'Connection': 'keep-alive'
            }
        #初始化存储类
        # self.saver=Saver()
    def _run_one_page(self,page_id):
        #获得页面的url地址
        url = SpiderConfig.base_url + str(page_id) + ".htm"
        try:
            my_request = request.Request(url, headers=self.headers)
            my_response = request.urlopen(my_request)
        except:
            print("页面出错！")
        else:
            print(url)
            self.saver.save_one_item(sp.get_one_page(url, page_id))

    def get_one_page_data(self,url):
        try:
            my_request = request.Request(url, headers=self.headers)
            my_response = request.urlopen(my_request)
        except:
            print("页面出错！")
        else:
            print(url)
            item = sp.get_one_page(url,5000)
            return item

    def run_spider(self,page):
        self._run_one_page(page)

if __name__ == '__main__':
    r = Runner()
    r.get_one_page_data("https://baike.baidu.com/item/%E7%99%BE%E5%BA%A6%E7%99%BE%E7%A7%91/85895?fr=aladdin")