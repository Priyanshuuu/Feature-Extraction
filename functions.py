import pandas as pd
import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup


def Date(text):
    date_list = []
    date=text.find_all("div", {"class","item-label"})      
    l = len(date)
    for contsD in range(l) :
        contyD = date[contsD].text
        date_list.append(contyD)
    return date_list   

def link_opener(text):
    my_urllink = text
    uClient = uReq(my_urllink)
    page_html = uClient.read()
    uClient.close()
    page_soup = soup(page_html,"html.parser")
    return page_soup

def cat_li1(listy):
    li1 = []
    li1.clear()
    for i in range(len(listy)):
        li1.append("misc")
    return li1   
            
def cat_li2(listy):
    li2 = []
    li2.clear()
    for i in range(len(listy)):
        li2.append("data_breach")
    return li2   

def cat_li3(listy):
    li3 = []
    li3.clear()
    for i in range(len(listy)):
        li3.append("Vulnerabilities")
    return li3   

def nextpag(text):
    next_pages = text.find_all("a",{"class","blog-pager-older-link-mobile"})
    print(next_pages[0]['href'])
    y = next_pages[0]['href']
    nextp = link_opener(y)
    return nextp
"""
def link_finder(text):
    linky=[]
    linkytext=[]
    while True:
        try:
            linknext = text.find_all("a",{"class","blog-pager-older-link-mobile"})[0]['href']
            for i in range(len(linknext)):
            while linknext[i]!=0:
                  z=nextpag(text)
                  linky.append(z)
                  zx=article(z)
                  linkytext.append(zx)
        
                  if linknext==0:
                     break
    return linky,linkytext   
"""    
def article(text):
    lisssty=[]
    link=[]
    pag_arc=text.find_all("a",{"class","story-link"})
    for index in range(len(pag_arc)):
        pag_arc_link=pag_arc[index]['href']
        link.append(pag_arc_link)
    for index2 in range(len(link)):
        pag_arc_soup=link_opener(link[index2])
        pag_arc_para=pag_arc_soup.find_all("div",{"class","articlebody clear cf"})[0].text        
        lisssty.append(pag_arc_para)
        print("article Scrapped")
    return lisssty  


def link_finder2(text):
    linky=[]
    #linkytext=[]
    while True:
            linknext = text.find_all("a",{"class","blog-pager-older-link-mobile"})
            
            if linknext==[]:
                #pass # list is empty
                print("last page reached")
                break
            else:
                #print(linknext)
                li=linknext[0]['href']
                #print(li)
                linknext_soup=link_opener(li)
                text=linknext_soup
                linky.append(li)
    return linky
def token(text):
    tok = re.split('\W+',text)
    return(tok)
    
    
def rem(tokens):
    stop=[word for word in tokens if word not in stopword]
    return stop


def stemmin(tokens):
    text = [ps.stem(word) for word in tokens]
    return text



def lemma(tokens):
    text = [wn.lemmatize(word) for word in tokens]
    return text
    

def headline(text):
    headlines_list=[]
    headlines = text.find_all("h2", {"class","home-title"})      
    l=len(headlines)
    
    for conts in range(l) :
        conty= headlines[conts].text
        headlines_list.append(conty)
        print("headline bhi hogyai")
    
    return headlines_list

def headlinks(text):
    headlink_list=[]
    headlink=text.find_all("a",{"class","story-link"})      
    l=len(headlink)
    for contshl in range(l) :
        contyhl= headlink[contshl]['href']
        headlink_list.append(contyhl)
    return headlink_list 

