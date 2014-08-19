
# coding: utf-8

# In[2]:

import numpy as np
import string
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import sklearn
from sklearn import linear_model
from scipy import sparse
from scipy import linalg
from sklearn.datasets.samples_generator import make_regression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pylab import *
import pylab
import ggplot
from selenium.webdriver.common.keys import Keys
import time
import mpld3
import seaborn


# In[3]:

# listings is file object with a row per address
# this scrapes redfin details page per address and writes the result to training_file
# note: this writes to the file one row at a time in case the scrape fails mid-way
def createtrainingset(listings, training_file, crime_):
    
    browser = webdriver.Firefox()
    browser.get('http://www.redfin.com/CA/Burlingame/3077-Mariposa-Dr-94010/home/2048468')
    time.sleep(35) #this gives you a chance to log into the website :p
    
    outf = open(training_file,'a')
    line = listings.readline()
    while line:
        listing = line.split('\t')
        if listing[0]=='Past Sale':
            home_id = listing[2]
            len(listing)
            redfin_url = listing[24]
            lat, lon = (listing[-3] , listing[-2])
            year_built, zip_code, list_price, beds, baths, sqft, dom, parking, orig_list_price = (
                listing[12] , listing[5], listing[6], listing[7],
                listing[8], listing[10], listing[15], listing[13], listing[21]
                )
            browser.get(redfin_url)
            t=browser.page_source
            l=browser.find_elements_by_class_name('as-data-section')
            redfin_vars = [x.text for x in l]
            #gd=browser.find_elements_by_id('propertyDetailsId')
            #details = [x.text for x in gd]
            #highschool = ','.join(details).split('\n')[241]
            
            counter = 0
            if abs(float(lat)) > 1:
                g = [float(lat),float(lon)]
                for k in range(len(crime_)):
                    if abs(abs(g[0])-abs(float(crime_[k][1]))) < 0.0047 and abs(abs(g[1])-abs(float(crime_[k][0]))) < 0.0037: 
                        counter = counter + float(crime_[k][2])
            if len(redfin_vars) == 4 and len(redfin_vars[0]) > 0: #checks for active listings
                def get_num(x):
                    return x.split('\n')[0].split()[0].replace(',','')
                views = get_num(redfin_vars[0])
                favs = get_num(redfin_vars[1])
                xouts = get_num(redfin_vars[2])
                home_summary = [home_id, year_built, zip_code, list_price, beds, 
                                baths, sqft, dom, parking, orig_list_price, views, 
                                favs, xouts, lat, lon, counter]
                outf.write(','.join(str(e) for e in home_summary)+ "\n")
        line = listings.readline()
    listings.close()
    outf.close()


# In[4]:

def parse_text_rows(data, walk, filter_lots=True): 
    out_data = []
    for line in data:
        d = {}
        line_parts = line.split(',')

        
        if len(line_parts) < 6:
            continue
        
        if line_parts[3]:
            if int(line_parts[3]) < 100000:
                continue
                    
        if filter_lots:
            if not line_parts[6]:
                continue
        
        d['home'] = line_parts[0]
        
        if (line_parts[1]):
            if int(line_parts[1]) > 1400:
                d['year_built'] = int(line_parts[1])
            else:
                continue
        else:
            continue
            
        if line_parts[2]:
            d['zipcode'] = (line_parts[2])
        else:
            continue

        if line_parts[3] and line_parts[6]:
            d['sale_price'] = int(line_parts[3])
            d['sqft'] = int(line_parts[6])
            d['_price_per_sqft'] = (1.0*int(line_parts[3])/int(line_parts[6]))
            if line_parts[9]:
                d['list_price'] = (1.0*int(line_parts[9])/int(line_parts[6]))
            else:
                continue
        else:
            continue
            
        if line_parts[4]:
            d['beds'] = int(line_parts[4])
        else:
            continue

        if line_parts[5]:
            d['baths'] = float(line_parts[5])
        else:
            continue
            
        if line_parts[8]:
            if int(line_parts[8]) < 10:
                d['parking'] = int(line_parts[8])
            else:
                continue
        else:
            continue
            
        if line_parts[9]:
            d['list_price'] = int(line_parts[9])
        else:
            continue
            
        if line_parts[10]:
            d['views'] = int(line_parts[10])
        else:
            continue
        
        if int(line_parts[10]) > 0:
            if 1.0*int(line_parts[11])/int(line_parts[10]) < 0.7:
                d['fav_per_view'] = 1.0*int(line_parts[11])/int(line_parts[10])
            else:
                continue
        else:
            continue
            
        if int(line_parts[10]) > 0:
            if 1.0*int(line_parts[12])/int(line_parts[10]) < 0.7:
                d['xout_per_view'] = 1.0*int(line_parts[12])/int(line_parts[10])
            else:
                continue
        else:
            continue
        
        if (line_parts[15]) > 0:
            d['crime_score'] = float(line_parts[15])
        else:
            continue

        L = len(walk_contents)
        sim_val = 1000
        for i in range(L):
            #latlong_diff = abs(float(walk[i].split(',')[-2])-float(line_parts[-3])) + abs(float(walk[i].split(',')[-1])-float(line_parts[-2]))
            latlong_diff = sqrt((float(walk[i].split(',')[-2])-float(line_parts[-3]))**2 + (float(walk[i].split(',')[-1])-float(line_parts[-2]))**2)
            if latlong_diff < sim_val:
                sim_val = latlong_diff
                ind = i
        
        d['latitude'] = float(line_parts[13]) + (1e-4)*(rand())
        d['longitude'] = float(line_parts[14]) + (1e-4)*(rand())
        
        if line_parts[7]:
            d['dom'] = int(line_parts[7])
        else:
            d['dom'] = 0
            
        if line_parts[12]:
            d['xouts'] = int(line_parts[12])
        else:
            d['xouts'] = 0
            
        if line_parts[11]:
            d['favs'] = int(line_parts[11])
        else:
            d['favs'] = 0

        d['walk_score'] = int(walk[ind].split(',')[2]) 
        d['transit_score'] = int(walk[ind].split(',')[3]) 
        d['bike_score'] = int(walk[ind].split(',')[4])   

        out_data.append(d)
    return out_data


# In[5]:

def linearregfit(x,y):
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    rcoeff = regr.coef_
    rscore = regr.score(x,y)

    #Calculate standard error for each coeffiecient (code from stackoverflow question 20938154)
    MSE = np.mean((y - regr.predict(x).T)**2)
    var_est = MSE * np.diag(np.linalg.pinv(np.dot(x.T,x)))
    SE_est = np.sqrt(var_est)

    return rcoeff, rscore, SE_est


# In[6]:

if __name__=='__main__':

    OBTAIN_TRAINING_SET = False
    PLOTTER = False
    LOGPLOTTER = False
    
    raw_data = 'SF_listings.txt' #input list of addresses
    crime_file = 'clustered_crime_data.csv' #from data.sfgov.org (clustered with kmeans in Matlab)
    walk_score_file = 'walkscore_full.txt'
    
    training_file = 'SF_past_listings_minus_test.txt' #(to be) decorated with scraped data

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
    data = trainingset.readlines()
    trainingset.close()
    
    walk_scores = open(walk_score_file,'r')
    walk_contents = walk_scores.readlines()
    close(walk_score_file)
    # unique-ify, note address is in each line
    data = list(set([item.strip() + '\n' for item in data]))
    
    # turn data into a list of dicts, [ {year built:..., price per sqft: ...}, ]
    data = parse_text_rows(data,walk_contents)

    df_ = pd.DataFrame(data)
    
    #dataframes to compare
    
    data_for_plotting = df_.drop(['favs','xouts','dom','latitude','longitude','zipcode','sale_price','list_price','home'],axis=1)
    data_with_redfinvars = df_.drop(['favs','xouts','dom','latitude','longitude','_price_per_sqft','zipcode','sale_price','home'],axis=1)
    data_without_redfinvars = data_with_redfinvars.drop(['fav_per_view','xout_per_view','views'],axis=1)
    redfin_only = data_without_redfinvars.drop(['walk_score','transit_score','bike_score','crime_score'],axis=1)
    simple_plotting = df_.drop(['favs','xouts','dom','latitude','longitude','fav_per_view','baths','beds','sqft','zipcode','sale_price','parking',
                                'list_price','walk_score','bike_score','crime_score','views','home'],axis=1)



# In[7]:

get_ipython().magic(u'matplotlib inline')
font = {'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)


# In[8]:

pylab.figure()

n, bins, patches = pylab.hist(df_['sale_price'], 30, histtype='barstacked')
plt.xlabel('Sale Price')
plt.ylabel('Number of Homes')
savefig("Sale price histograme",format='png')
fig = gcf()
html = mpld3.fig_to_html(fig)
pylab.show()


## Extract a new test sets from the training set:

# In[9]:

DIVY = False

if DIVY:    
    fa = np.ones(300)

    for i in range(300):
        fa[i] = int((len(data) - 0)*rand())
    
    training_file = 'SF_past_listings_scraped.txt' #decorated with scraped data
    trainingset = open(training_file,'r')
    
    new_test = open('SF_past_listings_scraped_test_4.txt','a')
    training_minus_test = open('SF_past_listings_minus_test.txt','a')

    
    data = trainingset.readlines()
    trainingset.close()
    for j in range(len(data)):
        if j in fa:
            new_test.write(data[j])
        else:
            training_minus_test.write(data[j])


## Analysis:

# In[10]:

if PLOTTER:
    scatter_matrix(data_for_plotting, alpha=0.2, diagonal='kde',figsize=(15,11))
    savefig("ScatterMatrix.png",format='png')
    plt.show()


# In[11]:

if LOGPLOTTER:
    axl = scatter_matrix(data_for_plotting, alpha=0.2, diagonal='kde',figsize=(15,11))
    for i, axs in enumerate(axl):
        for j, ax in enumerate(axs):
            ax.set_xscale('log')
            #ax.set_yscale('log')
    savefig("SemilogScatterMatrix.png",format='png')
    plt.show()


## Multiple Linear Regression Analysis:

# In[12]:

print('MULTIPLE LINEAR REGRESSION WITHOUT DECORATED VARIABLES')
print('-------------------------------------------------------------')
print(data_without_redfinvars.columns)
(rcoeff, rscore, SE_est) = linearregfit(redfin_only,df_.ix[:,0])
print("Regression Coefficients")
print(rcoeff)
print("Standard Error for each coefficient")
print(SE_est)
print("Regression Score")
print(rscore)
print('')

print('MULTIPLE LINEAR REGRESSION INCLUDING DECORATED VARIABLES')
print('-------------------------------------------------------------')
print(data_with_redfinvars.columns)
(rcoeff, rscore, SE_est) = linearregfit(data_with_redfinvars,df_.ix[:,0])
print("Regression Coefficients")
print(rcoeff)
print("Standard Error for each coefficient")
print(SE_est)
print("Regression Score")
print(rscore)
print('')


## Calculate Confidence Intervals (95%) for MLR

# In[13]:

confid_int_upper = [0]*len(SE_est)
for i in range(len(SE_est)):
    confid_int_upper[i] = list(rcoeff)[i]+(list(SE_est)[i]*1.96) 

rcoeff_list = list(rcoeff)
confid_int_list = list(confid_int_upper)
fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot(2,1,2)
ax1.set_ylim([-1,len(rcoeff)+1])
plt.errorbar(rcoeff_list, range(len(rcoeff_list)), xerr=confid_int_list,
    linestyle='None', marker="o", color="purple",
    markersize=5, linewidth=1.75, capsize=20
)
group_labels = data_with_redfinvars.columns
plt.yticks(np.arange(len(rcoeff_list)))
ax1.set_yticklabels(group_labels)
plt.xlabel('Coefficient Confidence Intervals (95%)')
savefig("confindence_intervals_full.png",format='png')

plt.show()
del(rcoeff_list[11])
del(confid_int_list[11])
del(rcoeff_list[4])
del(confid_int_list[4])


ax2 = plt.subplot(1,2,2)

ax2.set_ylim([-1,len(rcoeff)+1])

plt.errorbar(rcoeff_list,range(len(rcoeff_list)),xerr=confid_int_list,
    linestyle='None',marker="o", color="purple",
    markersize=5,linewidth=1.75,capsize=20
)
group_labels = ['Baths','Beds','bike_score','crime_score','list_price','parking','sqft','transit_score','views','walk_score','year_built']
plt.yticks(np.arange(10))
ax2.set_yticklabels(group_labels)
plt.xlabel('Coefficient Confidence Intervals (95%)')
savefig("confindence_intervals.png",format='png')
plt.show()


## RF Regression for data WITH decorated variables:

# In[15]:

clf = RandomForestRegressor(n_estimators=1000,max_features=12, verbose=0)
clf = clf.fit(data_with_redfinvars,df_.ix[:,0])
G = clf.feature_importances_


for i in range(len(data_with_redfinvars.columns)):
    print [data_with_redfinvars.columns[i], G[i]]
#year built, walk score, transit score, xout_per_view, views, 
score = clf.score(data_with_redfinvars,df_.ix[:,0])
print(['R2',score])

test = open('SF_past_listings_scraped_test_4.txt','r')
test_contents = test.readlines()
test_data = parse_text_rows(test_contents,walk_contents)

close('SF_past_listings_scraped_test_4.txt')
df_test = pd.DataFrame(test_data)
df_compare = pd.DataFrame(test_data)


df_test = df_test.drop(['favs','xouts','dom','latitude','longitude','home','zipcode','sale_price','_price_per_sqft'],axis=1)
est = clf.predict(df_test)

error_array = []
sale_list_array = []
for i in range(len(test_data)):
    err = abs(est[i] - df_compare['_price_per_sqft'][i])
    err2 = (est[i] - df_compare['_price_per_sqft'][i])
    temp = err2/df_compare['_price_per_sqft'][i]
    percent_error = err/df_compare['_price_per_sqft'][i]
    if percent_error > 1:
        print [ df_compare['zipcode'][i], df_compare['sale_price'][i]]
    error_array.append(percent_error*100)
    sale_list_array.append(temp)
print(['percent error',mean(error_array)])
print(['median error',median(error_array)])
print(['n',len(df_)])


# In[16]:

pylab.figure()

n, bins, patches = pylab.hist(error_array, 30, histtype='barstacked')
plt.xlabel('Percent Error in Predictions on Blind Test Set')
plt.ylabel('Number of Homes')
savefig("percent_error_in_predictions",format='png')
fig = gcf()
html = mpld3.fig_to_html(fig)
open("percenterror_d3.html", 'w').write(html)
pylab.show()
mean(error_array)


# In[17]:

within5 = 0
within10 = 0
within15 = 0
for i in range(len(error_array)):
    if error_array[i]/100 <= .05:
        within5 = within5 + 1
        within10 = within10 + 1
        within15 = within15 + 1
    else:
        if error_array[i]/100 <= .1:
            within10 = within10 + 1
            within15 = within15 + 1
        else:
            if error_array[i]/100 <= .15:
                within15 = within15 + 1
print (1.0*within5/len(error_array), 1.0*within10/len(error_array), 1.0*within15/len(error_array))


# In[60]:

N = len(G)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots(figsize=(20,10))
rects1 = ax.bar(ind, sorted(G,reverse=True), width, color='m')
plt.xticks(np.arange(len(G)))

ax.set_ylabel('Scores')
ax.set_title('Relative importance of vars')
ax.set_xticklabels(['transit score','year built','list price','walk score','sqft','fav_per_view',
                    'bike score','crime score','views','beds','xouts_per_view',
                    'baths','parking'])
savefig("var_rank.png",format='png')
plt.show()


## Test the model in absence of some parameters

# In[18]:

TEST_MODEL = False

if TEST_MODEL:

    clf = RandomForestRegressor(n_estimators=1000)

    data_missing_stuff = data_with_redfinvars.drop(['list_price'
                                                    ],axis=1)
    clf = clf.fit(data_missing_stuff,df_.ix[:,0])
    G = clf.feature_importances_


    #year built, walk score, transit score, xout_per_view, views, 
    score = clf.score(data_missing_stuff,df_.ix[:,0])
    print(['R2',score])

    test = open('SF_past_listings_scraped_test_4.txt','r')
    test_contents = test.readlines()
    test_data = parse_text_rows(test_contents,walk_contents)

    close('SF_past_listings_scraped_test_2.txt')
    df_test = pd.DataFrame(test_data)
    df_compare = pd.DataFrame(test_data)


    df_test = df_test.drop(['list_price',
                            'favs','xouts',
                            'dom','latitude','longitude','home',
                            'zipcode','sale_price','_price_per_sqft'],axis=1)
    est = clf.predict(df_test)
    print df_test.columns
    error_array = []
    sale_list_array = []
    for i in range(len(test_data)):
        err = abs(est[i] - df_compare['_price_per_sqft'][i])
        err2 = (est[i] - df_compare['_price_per_sqft'][i])
        temp = err2/df_compare['_price_per_sqft'][i]
        percent_error = err/df_compare['_price_per_sqft'][i]
        error_array.append(percent_error*100)
        sale_list_array.append(temp)
    print(['percent error',mean(error_array)])
    print(['median error',median(error_array)])


    within5 = 0
    within10 = 0
    within20 = 0
    for i in range(len(error_array)):
        if error_array[i]/100 <= .05:
            within5 = within5 + 1
            within10 = within10 + 1
            within20 = within20 + 1
        else:
            if error_array[i]/100 <= .1:
                within10 = within10 + 1
                within20 = within20 + 1
            else:
                if error_array[i]/100 <= .2:
                    within20 = within20 + 1
    print (1.0*within5/len(error_array), 1.0*within10/len(error_array), 
           1.0*within20/len(error_array))


## Print Predictions to file

# In[19]:

SF_current_listings = open('SF_new_listings_625_scraped.txt','r')
SF_current_listings_contents = SF_current_listings.readlines()
test_data = parse_text_rows(SF_current_listings_contents,walk_contents)
close('SF_new_listings_625_scraped.txt')
df_current = pd.DataFrame(test_data)
df_current_compare = pd.DataFrame(test_data)

dbfile = open('SF_current_listings_predicted_625_staggered.csv','a')

df_current = df_current.drop(['favs','xouts','dom','latitude','longitude','home','zipcode','sale_price','_price_per_sqft'],axis=1)

est = clf.predict(df_current)

PRINT_DB = True



if PRINT_DB:
    for i in range(len(test_data)):
        
        G = [df_current_compare['home'][i], df_current_compare['year_built'][i], 
             df_current_compare['zipcode'][i],df_current_compare['list_price'][i],
             df_current_compare['beds'][i], df_current_compare['baths'][i],
        df_current_compare['sqft'][i],df_current_compare['dom'][i],
        df_current_compare['parking'][i], df_current_compare['sale_price'][i],
                df_current_compare['views'][i], df_current_compare['favs'][i],
                        df_current_compare['xouts'][i], df_current_compare['latitude'][i],
                                                df_current_compare['longitude'][i], df_current_compare['crime_score'][i]]



        diff = 100*((est[i]*df_current_compare['sqft'][i]-
                     df_current_compare['list_price'][i])/(df_current_compare['list_price'][i]))

        
        if diff >= 5:
            color = "green"
        if diff > -5 and diff < 5:
            color = "orange"
        if diff <= -5:
            color = "red"
        
        G.extend([est[i],df_current.ix[i,2],df_current.ix[i,8],df_current.ix[i,10],color])

        dbfile.write(','.join(str(e) for e in G)+ "\n")


# In[20]:

difference_array = []
percent_difference_array = []
list_sale_array = []
five=0
afive=0
bfive=0
for i in range(len(df_)):
    diff = df_['sale_price'][i] - df_['list_price'][i]
    pdiff = 100*((diff*1.0)/df_['sale_price'][i])
    if abs(pdiff) <5:
        five = five + 1
    else:
        if pdiff >=5:
            afive = afive + 1
        else:
            if pdiff <= -5:
                bfive = bfive + 1
    if abs(pdiff) < 90:
        difference_array.append(diff)
        percent_difference_array.append(pdiff)


# In[21]:

pylab.figure()

n, bins, patches = pylab.hist(difference_array
                                , 75, histtype='barstacked')
plt.xlabel('sale price minus list price')
plt.ylabel('Number of Homes')
savefig("difference_array",format='png')
pylab.show()


# In[22]:

pylab.figure()

n, bins, patches = pylab.hist(sort(percent_difference_array)[1:150], 22, histtype='barstacked',color=['crimson'])
n, bins, patches = pylab.hist(sort(percent_difference_array)[151:730], 5, histtype='barstacked',color=['orange'])
n, bins, patches = pylab.hist(sort(percent_difference_array)[731:], 20, histtype='barstacked',color=['Chartreuse'])

plt.xlabel('Sale Price as Percent Above Asking Price')
plt.ylabel('Number of Homes')
savefig("percent_difference_array.png",format='png')
pylab.show()

