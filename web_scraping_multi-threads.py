import requests
from bs4 import BeautifulSoup
import json
import time 
start=time.perf_counter()
import concurrent.futures

URL = "https://www.cermati.com/karir"
r = requests.get(URL)
URL_LIST=[]
page_soup = BeautifulSoup(r.content, 'html5lib')


def parell_tasks(url):
    temp=[]
    description = []
    qualification = []
    category,link=url.split('|')

    r = requests.get(link)
    page_soup = BeautifulSoup(r.content, "html5lib")
    title = str(page_soup.find("h1", {"class": "job-title"}).text.strip())
    location = str(page_soup.find("span", {"class": "job-detail"}).text.strip())
    for a in page_soup.findAll("div", {"class": "wysiwyg", "itemprop": "responsibilities"}):
        for l in a.findAll("li"):
            description.append(l.text)
        if not description:
            for l in a.findAll("p"):
                if "Walk-in interview" in l.text:
                    break
                else:
                    description.append(l.text)
    for a in page_soup.findAll("div",
                               {"class": "wysiwyg", "itemprop": "qualification", "itemprop": "qualifications"}):
        for l in a.findAll("li"):
            qualification.append(l.text)
        if not qualification:
            for l in a.findAll("p"):
                if "Walk-in interview" in l.text:
                    break
                else:
                    qualification.append(l.text)
        posted_by = 'None'
    try:
        posted_by = str(page_soup.find("h3", {"class": "details-title"}).text.strip())
    except:
        pass
    try:
        temp.append({"title": title, "location": location, "description": description, "qualification": qualification,
                     "posted by": posted_by})
    except:
        posted_by = 'None'
        temp.append({"title": title, "location": location, "description": description, "qualification": qualification,
                     "posted by": posted_by})
    return category,temp




for tab in page_soup.findAll("div", {"tab-pane"}):

    category = tab.find("h4", {"tab-title"}).text.strip()
    temp = []
    for link in tab.findAll("a"):
        
        my_url = link.get("href")
        URL_LIST.append(category+'|'+my_url)

with concurrent.futures.ProcessPoolExecutor() as executor:

    result=[executor.submit(parell_tasks,url) for url in URL_LIST]
    for f in concurrent.futures.as_completed(result):
        print(f.result())
