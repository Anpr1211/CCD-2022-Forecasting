# command to install the YAKE library for keyword extraction
# !pip install git+https://github.com/LIAAD/yake

# libraries
import json      
import pandas as pd
import yake
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("Script begin")

# read the dictionary of data source
f = open('data_source.json')
data_source = json.load(f)

# read raw data
data = pd.read_csv(data_source["local_address"])

print("Data loaded.")

# removing personal info (name, phone number, email, etc.)
personal_info =["phone", "photo_url", "email", "self_phone", "self_email", "self_name"]  # not removing name
data.drop(personal_info, axis=1, inplace=True)

# convert profiles to binary variables
data.insert(2, "blog_binary", pd.notnull(data.blog).replace({True: 1, False: 0}))
data.drop("blog", axis=1, inplace=True)

data.insert(3, "github_binary", pd.notnull(data.github).replace({True: 1, False: 0}))
data.drop("github", axis=1, inplace=True)

data.insert(4, "linkedin_binary", pd.notnull(data.linkedin).replace({True: 1, False: 0}))
data.drop("linkedin", axis=1, inplace=True)

data.insert(6, "about_binary", pd.notnull(data.about).replace({True: 1, False: 0}))

# removing name and college from About

for i in range(len(data)):
    
    names = data.name[i].split() # split full name since some have written just their first names in About
    
    for n in names:
        if data.about_binary[i]==1 and n in data.about[i]:
            data.about[i] = data.about[i].replace(n, "")      # check and remove names
            
    if data.organization[i]==np.nan and data.organization[i] in data.about[i]:   # check and remove college names
        data.about[i] = data.about[i].replace(data.organization[i], "")

data.drop("name", axis=1, inplace=True)        # removing name column

# defining the extractor for 5 keywords
kw_extractor = yake.KeywordExtractor(top=5, stopwords=None)

keywords = [None] * len(data) # empty list to store keywords

# extracting keywords
for i in tqdm(range(len(data))):
    if data.about_binary[i] == 1:
        text = data.about[i]        
        
        keywords[i] = kw_extractor.extract_keywords(text)
    
data.insert(4, "about_keywords", keywords) # adding the keywords to the dataframe
data.drop("about", axis=1, inplace=True)   # removing the about column

# removing duplicates
reviewers = ['Reviewer 1', 'Reviewer 2', 'Reviewer 3']   # list of reviewer columns
rows_to_be_deleted = [] # list to store the index of rows to be deleted

# finding duplicates
for r in reviewers:
    
    data[r] = data[r].astype(str)
    for i in range(len(data)):
      
        if "Duplicate" in data[r][i]:
            rows_to_be_deleted.append(i)
        
# removing rows 
data.drop(rows_to_be_deleted, inplace=True)
data.reset_index(inplace=True)

print("Processing complete.")
# save file 
data.to_csv("clean_data.csv", index=False)

print("Written to file. Exit.")