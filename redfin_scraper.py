
# coding: utf-8

# In[8]:

import numpy as np
import string
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import sklearn
from sklearn import linear_model
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pylab import *
import ggplot
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
import time
from re import sub
from decimal import Decimal


# In[16]:

# listings is file object with a row per address
# this scrapes redfin details page per address and writes the result to training_file
# note: this writes to the file one row at a time in case the scrape fails mid-way
def createtrainingset(listings, training_file, crime_):
    outf = open(training_file,'a')
    line = listings.readline()
    while line:
        listing = line.split('\t')
        if listing[0]=='MLS Listing':
        #if listing[0]=='Past Sale' or listing[0]=='MLS Listing':
            home_id = listing[2]
            redfin_url = listing[24]
            lat, lon = (listing[-3] , listing[-2])
            year_built, zip_code, list_price, beds, baths, sqft, dom, parking, orig_list_price = (
                listing[12] , listing[5], listing[6], listing[7],
                listing[8], listing[10], listing[15], listing[13], listing[21]
                )
            
            if listing[10] and listing[7] and listing[8] and listing[12] and listing[-2] and listing[-3]:
                browser.get(redfin_url)
                t=browser.page_source
                l=browser.find_elements_by_class_name('as-data-section')

                redfin_vars = [x.text for x in l]
                gd=browser.find_elements_by_id('propertyDetailsId')
                details = [x.text for x in gd]

                if len(','.join(details).split('\n')) >= 2:
                    lsp = ','.join(details).split('\n')[2]
                    lastsoldprice = Decimal(sub(r'[^\d.]', '', lsp))
                else:
                    lastsoldprice = ''

                counter = 0
                if abs(float(lat)) > 1:
                    g = [float(lat),float(lon)]
                    for k in range(len(crime_)):
                        if abs(abs(g[0])-abs(float(crime_[k][1]))) < 0.0047 and abs(abs(g[1])-abs(float(crime_[k][0]))) < 0.0037: 
                            counter = counter + float(crime_[k][2])
                if len(redfin_vars) == 4 and len(redfin_vars[0]) > 0: #checks for complete listings
                    def get_num(x):
                        return x.split('\n')[0].split()[0].replace(',','')
                    views = get_num(redfin_vars[0])
                    favs = get_num(redfin_vars[1])
                    xouts = get_num(redfin_vars[2])
                    home_summary = [home_id, year_built, zip_code, lastsoldprice, beds, 
                                    baths, sqft, dom, parking, orig_list_price, views, 
                                    favs, xouts, lat, lon, counter]
                    outf.write(','.join(str(e) for e in home_summary)+ "\n")

        line = listings.readline()
    listings.close()
    outf.close()


# In[20]:

if __name__=='__main__':

    browser = webdriver.Firefox()
    browser.get('http://www.redfin.com/CA/Burlingame/3077-Mariposa-Dr-94010/home/2048468')
    time.sleep(25) 
    
    OBTAIN_TRAINING_SET = True
    
    raw_data = 'SF_new_listings_625_chunked.txt' #input list of addresses
    training_file = 'SF_new_listings_625_scraped.txt' #decorated with scraped data
    
    crime_file = 'clustered_crime_data.csv' #from data.sfgov.org (clustered with kmeans in Matlab)
    crime_data = open(crime_file,'r')
    crime_contents = crime_data.readlines()
    crime_strip = [x.split(',') for x in crime_contents[1:len(crime_contents)]]
    close(crime_file)
    
    if OBTAIN_TRAINING_SET:
        homelistings = open(raw_data, "r")
        # per house, scrape data and write to file incrementally
        createtrainingset(homelistings, training_file,crime_strip)
        homelistings.close()

    trainingset = open(training_file,'r')


# In[ ]:



