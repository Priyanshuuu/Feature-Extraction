# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:12:27 2019

@author: ishan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 05:22:38 2019

@author: ishan
"""
import trendfunc
import pandas as pd
soup_data = trendfunc.link_opener('https://blog.trendmicro.com/category/vulnerabilities/')
import warnings
warnings.filterwarnings(action='ignore')


# headings extraction
topic_list=[]
topic_link_list=[]
first_cat1=[]
sec_cat1=[]
thr_cat1=[]
first_cat1_dates=[]
sec_cat1_dates=[]
thr_cat1_dates=[]
cat=[]
dates=[]
articles=[]

topics=soup_data.find_all("li", {"class","menu-item menu-item-type-taxonomy menu-item-object-category current-menu-item"})
l=len(topics)
url='https://thehackernews.com/'
for topic in range(l) :
    topicy= topics[topic].a.text
    topic_list.append(topicy)
    topic_list=topic_list[0:1]
    link=topics[topic].a['href']
    topic_link_list.append(link)
    masterl=topic_link_list[0:1]
    for topicl in range(len(masterl)):
        if topicl==0:
            first_cat1.append(trendfunc.link_opener(masterl[topicl]))

master=[]  
dic={}
dic2={}
dic3={}
gen_list = [first_cat1[0]]


    
for ge in range(len(gen_list)):
    if ge==0:
        # print(trendfunc.headline(gen_list[ge]))
        # print(trendfunc.Date(gen_list[ge]))
        #fpage02 = trendfunc.nextpag(gen_list[ge])
        # print(trendfunc.headline(fpage2))
        # print(trendfunc.Date(fpage2))
        #fpage03 = trendfunc.nextpag(fpage02)
        # print(trendfunc.headline(fpage3))
        # print(trendfunc.Date(fpage3))        
        
        dic["arcticle_text1"]=trendfunc.article(gen_list[ge])
        #dic["arcticle_text2"]=trendfunc.article(fpage02)
        #dic["arcticle_text3"]=trendfunc.article(fpage03)
        
        
        #dic["head1"] = trendfunc.headline(gen_list[ge])
        #dic["date1"] = trendfunc.Date(gen_list[ge])
        #dic["Cat1"]=trendfunc.cat_li3(trendfunc.headline(gen_list[ge]))
        #for he in range(len(dic["head1"])):
         #   dic.setdefault(cat, []).append("Misc")
        #trendfunc.cat_pri(dic,"head1",topic_list)
        #dic["head2"] = trendfunc.headline(fpage02)
        #dic["date2"] = trendfunc.Date(fpage02)
        #dic["Cat2"]=trendfunc.cat_li3(trendfunc.headline(gen_list[ge]))
        #for he in range(len(dic["head2"])):
        #    dic["category"]="Misc"
        #trendfunc.cat_pri(dic,"head1",topic_list)    
        
        #dic["head3"] = trendfunc.headline(fpage03)
        #dic["date3"] = trendfunc.Date(fpage03)
        #dic["Cat3"]=trendfunc.cat_li3(trendfunc.headline(gen_list[ge]))
        #for he in range(len(dic["head3"])):
        #    dic["category"].append("Misc")
        #trendfunc.cat_pri(dic,"head1",topic_list)
        
        master.append(dic)
text_arc1=dic["arcticle_text1"]



