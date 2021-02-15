'''运行爬虫'''
from urllib import request
import requests
import numpy as np
import myspider.spider as sp
from myspider.run import Runner

if __name__ == '__main__':

    runner=Runner()

    for _ in range(10000):
        runner.run_spider(_)