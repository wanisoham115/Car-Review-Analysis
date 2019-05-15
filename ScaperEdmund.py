# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:06:06 2019

@author: Soham Wani
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from selenium import webdriver
from string import punctuation
import nltk
from nltk.corpus import stopwords
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
#%%

driver = webdriver.Chrome('C:/Users/AbPa/chromedriver.exe')
driver.get('https://forums.edmunds.com/discussion/2864/general/x/entry-level-luxury-performance-sedans/p715')
#Creating empty data frame to store user_id, dates and comments from ~5K users.
comments = pd.DataFrame(columns = ['Date','user_id','comments'])

j = 715
while (j>=1):
    # Running while loop only till we get 5K comments 
    if (len(comments)<5000):
        url = 'https://forums.edmunds.com/discussion/2864/general/x/entry-level-luxury-performance-sedans/p' + str(j)
        driver.get(url)
        ids = driver.find_elements_by_xpath("//*[contains(@id,'Comment_')]")
        comment_ids = []
        for i in ids:
            comment_ids.append(i.get_attribute('id'))

        for x in comment_ids:
            #Extract dates from for each user on a page
            user_date = driver.find_elements_by_xpath('//*[@id="' + x +'"]/div/div[2]/div[2]/span[1]/a/time')[0]
            date = user_date.get_attribute('title')

            #Extract user ids from each user on a page
            userid_element = driver.find_elements_by_xpath('//*[@id="' + x +'"]/div/div[2]/div[1]/span[1]/a[2]')[0]
            userid = userid_element.text

            #Extract Message for each user on a page
            user_message = driver.find_elements_by_xpath('//*[@id="' + x +'"]/div/div[3]/div/div[1]')[0]
            comment = user_message.text
            
            #Extracting Block Quote if Present
            block_quote = driver.find_element_by_xpath('//*[@id="' + x + '"]/div/div[3]/div/div[1]')
            block_quote_class = block_quote.find_elements_by_class_name('UserQuote')
            block_text = ''
            if len(block_quote_class)>0:
                block_text = block_quote_class[0].text
            
            #Replacing block quotes
            comment = comment.replace(block_text,"")
            
           #Adding date, userid and comment for each user in a dataframe    
            comments.loc[len(comments)] = [date,userid,comment]
        j=j-1
    else:
        break

#%%

comments_copy = copy.deepcopy(comments)

#removing /n from comments
def remove_space(s):
    return s.replace("\n"," ")

comments_copy['comments'] = comments_copy['comments'].apply(remove_space)
comments_copy.to_csv('comments.csv', header=True, sep=',')

#%%
#Cleansing the reviews we fetched from Edmunds.com

comments_copy = pd.read_csv('comments.csv')
models = pd.read_csv("models.csv", header = None, names = ['brand','model'], encoding='windows-1252')
comments_copy = comments_copy.dropna()
comments_copy.reset_index(inplace  = True)
comments_copy = comments_copy.drop(columns = ['index'])

#%%
def removepunc(item):
    for p in punctuation:
        item = item.lstrip().replace(p,'')
    return item

def lowerize(x):
    return x.lower()

comments_copy['comments_clean'] = comments_copy['comments'].apply(removepunc).apply(lowerize)
models['brand'] = models['brand'].apply(removepunc)
#%%
#replacing model names with manufacturer name
def model_to_brand(s):
    for i in models.index.values:
        s = s.replace(models["model"][i].lower(),models["brand"][i].lower())
    return s
comments_copy['comments_model_replace'] = comments_copy['comments_clean'].apply(model_to_brand)
#%%
comments_copy['comments_appear'] = comments_copy['comments_model_replace'].apply(word_tokenize).apply(set).apply(list)

# Remove stop words
stop_words = set(stopwords.words('english'))
def remove_stopwords(s):
    return [w for w in s if not w in stop_words] 
    
comments_copy['final_comments'] =  comments_copy['comments_appear'].apply(remove_stopwords)

#%%
count = []
for i in range(len(comments_copy)):
    count+=comments_copy['final_comments'][i]

#%%
    
from nltk import FreqDist
word_freq = nltk.FreqDist(count)

#%%

models_unique = models['brand'].drop_duplicates().tolist()

#%%
top_words = word_freq.most_common(500)

top_brands = []

for (key, items) in top_words:
    if key in models_unique:
        model_counts = (key,items)
        top_brands.append(model_counts)
        
#%%
top_15_brands_counts = top_brands[:15]
print ("\n\n *** Below are the top 15 brands according to their popularity *** \n\n" , top_15_brands_counts[:10])

#%%

top_15_brands_bar= pd.DataFrame(top_15_brands_counts)

label = top_15_brands_bar.loc[:,0]
freq = top_15_brands_bar.loc[:,1]

index = np.arange(len(label))
plt.bar(index, freq)
plt.xlabel('Brands', fontsize=12)
plt.ylabel('Frequencies', fontsize=12)
plt.xticks(index, label, fontsize=7, rotation=45)
plt.title('Top 15 brands ')
plt.show()


#%%

top_15_brands =[]
for brand, count in top_15_brands_counts:
    top_15_brands.append(brand)
    
#%%
new_df = pd.DataFrame(columns = top_15_brands)

def brand_mentioned(item):
    if brand in item:
        return 1
    else:
        return 0
      
for brand in top_15_brands:
    new_df[brand] = comments_copy['final_comments'].apply(brand_mentioned)

#%%
# Calculating lift among top brands
df2 = pd.DataFrame(columns = top_15_brands)

for i in range(len(top_15_brands)):
    new_list = []
    for j in range(len(top_15_brands)):
        if (i!=j):
            numerator = ((new_df[top_15_brands[i]] + new_df[top_15_brands[j]]) > 1).sum()
            denominator = new_df[top_15_brands[j]].sum()*new_df[top_15_brands[i]].sum()
            lift = numerator*len(new_df)/denominator
            df2.loc[top_15_brands[i],top_15_brands[j]] = lift
print ('Below are the lift ratios among top brands\n')
df2

#%%
attributes = pd.read_csv('attributes.csv')
#%%
def word_to_attributes(s):
    s = " ".join(str(x) for x in s)
    for i in attributes.index.values:
        s = s.replace(attributes["Attribute"][i].lower(),attributes["Mapping"][i].lower())
    return s
comments_copy['comments_attributes_replace'] = comments_copy['final_comments'].apply(word_to_attributes)
#%%
count = []
for i in range(len(comments_copy)):
    count+=comments_copy['comments_attributes_replace'][i]
attr_freq = nltk.FreqDist(count)
attributes_unique = attributes['Mapping'].drop_duplicates().tolist()

top_words = word_freq.most_common(3000)
top_attributes = []
for (key, items) in top_words:
    if key in attributes_unique:
        attribute_counts = (key,items)
        top_attributes.append(attribute_counts)
#%%
top_5_attributes_counts = top_attributes[:5]
print ('Below are the top 5 attributes \n' , top_5_attributes_counts)
#%%
# Fetching top 5 attrbutes
top_5_attributes =[]
for attribute, count in top_5_attributes_counts:
    top_5_attributes.append(attribute)
    
#%%
top_attributes
import matplotlib.pyplot as plt
 
values = [315,98,57,48,48,15,15]
colors = ['b', 'g', 'r', 'c', 'm','y']
labels = ['Performance', 'Maintenance', 'Styling', 'Comfort','Safety','Console','Efficiency']
explode = (0.1, 0, 0, 0, 0,0,0)
plt.pie(values, colors=colors, labels= labels,explode=explode,autopct='%1.1f%%',counterclock=False, shadow=True)
plt.title('Population Density Index')
plt.show()
#%%
attributes_df = pd.DataFrame(columns = top_5_attributes)

def attribute_mentioned(item):
    if attribute in item:
        return 1
    else:
        return 0
      
for attribute in top_5_attributes:
    attributes_df[attribute] = comments_copy['comments_attributes_replace'].apply(attribute_mentioned)
#%%
# Calculating Lift between top 5 brands and top 5 attributes
df3=pd.DataFrame(columns = top_5_attributes)
top_5_brands = top_15_brands[:5]
for i in range(len(top_5_brands)):
    new_list = []
    for j in range(len(top_5_attributes)):
        numerator = ((new_df[top_5_brands[i]] + attributes_df[top_5_attributes[j]]) > 1).sum()
        denominator = new_df[top_5_brands[i]].sum()*attributes_df[top_5_attributes[j]].sum()
        lift_brand_attributes = numerator*len(attributes_df)/denominator
        df3.loc[top_5_brands[i],top_5_attributes[j]] = lift_brand_attributes

print ('Below are the lift ratios between top 5 brands and top 5 attributes \n')
df3
#%%
def barchart(he,title):
    height = he
    bars = ('BMW', 'Audi', 'Acura', 'Mercedes', 'Honda')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
 
    # Create names on the x-axis
    plt.xticks(y_pos, bars)
    plt.title(title)
    # Show graphic
    plt.show()
    
#%%
barchart(df3.iloc[:,0],'Performance')

#%%
barchart(df3.iloc[:,1],'Maintenance')

#%%
barchart(df3.iloc[:,2],'Styling')

#%%
barchart(df3.iloc[:,3],'Comfort')

#%%
barchart(df3.iloc[:,4],'Safety')

#%%
aspiration = pd.read_csv("aspiration.csv")
#%%
def aspiring(s):
    #s = " ".join(str(x) for x in s)
    for i in aspiration['word'].index.values:
        s = s.replace(aspiration['word'][i],aspiration['aspr'][i])
    return s
comments_copy['comments_asp_replace'] = comments_copy['comments_attributes_replace'].apply(aspiring)
#%%
aspiring_df = pd.DataFrame(columns = ['aspiration'])

def aspiring_mentioned(item):
    if asp in item:
        return 1
    else:
        return 0

for asp in aspiration['aspr'].unique():
    aspiring_df[asp] = comments_copy['comments_asp_replace'].apply(aspiring_mentioned)
#%%
# Calculating Lift between top 5 brands and aspiration
aspiring_df2=pd.DataFrame(columns = ['aspiration'])
top_5_brands = top_15_brands[:5]
for i in range(len(top_5_brands)):
    new_list = []
    for j in range(len(aspiration['aspr'].unique())):
        numerator = ((new_df[top_5_brands[i]] + aspiring_df['aspiration']) > 1).sum()
        denominator = new_df[top_5_brands[i]].sum()*aspiring_df['aspiration'].sum()
        lift_brand_aspr = numerator*len(aspiring_df)/denominator
        aspiring_df2.loc[top_5_brands[i],'aspiration'] = lift_brand_aspr

print ('Below are the lift ratios between top 5 brands and aspiration \n')
aspiring_df2

#%%
import matplotlib.pyplot as plt
 
values = [1.44,1.85,2.44,2.12,1.84]
colors = ['b', 'g', 'r', 'c', 'm']
labels = ['BMW', 'Audi', 'Acura', 'Mercedes','Honda']
explode = (0, 0, 0.1, 0, 0)
plt.pie(values, colors=colors, labels= labels,explode=explode,autopct='%1.1f%%',counterclock=False, shadow=True)
plt.title('Population Density Index')
plt.show()

#%%
# Exporting the final cleaned data frame
final_df = comments_copy[['Date','user_id','comments','comments_asp_replace']]
final_df.columns = ['Date','user_id','comments','cleaned_comments']
final_df.to_csv('final_data_file.csv', header=True, sep=',')