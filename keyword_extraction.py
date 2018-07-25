# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:34:04 2018

@author: shweta
"""
import sys
sys.path 
import PyPDF2
import textract
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np

filepath ='JavaBasics-notes.pdf' 

#open file

File = open(filepath,'rb')
pdfRead = PyPDF2.PdfFileReader(File)


no_of_pages = pdfRead.numPages
count = 0
text = ""

#read each page
while count < no_of_pages:
    page = pdfRead.getPage(count)
    count +=1
    text += page.extractText()

if text != "":
   text = text

else:
   text = textract.process('http://bit.ly/epo_keyword_extraction_document', method='tesseract', language='eng')
text = text.lower()
tokens = word_tokenize(text)

#stopwords 
stop_words = stopwords.words('english')

keywords = [word for word in tokens if len(word)>1 and not word in stop_words and word.isalpha()]

#function to provide weightage to each keyword
def Keyword_weightage(word,text,no_of_documents=1):
    word_list = re.findall(word,text)
    word_appeared =len(word_list)
    tf_score = word_appeared/float(len(text))
    idf_score = np.log((no_of_documents)/float(word_appeared))
    tf_idf_together_score = tf_score*idf_score
    return word_appeared,tf_score,idf_score ,tf_idf_together_score 

#preparing dataframe and storing as csv 
df = pd.DataFrame(list(set(keywords)),columns=['Keywords'])
df['Word_appeared'] = df['Keywords'].apply(lambda x: Keyword_weightage(x,text)[0])
df['tf_score'] = df['Keywords'].apply(lambda x: Keyword_weightage(x,text)[1])
df['idf_score'] = df['Keywords'].apply(lambda x: Keyword_weightage(x,text)[2])
df['tf_idf_together_score'] = df['Keywords'].apply(lambda x: Keyword_weightage(x,text)[3])

df = df.sort_values('tf_idf_together_score',ascending=True)
df.to_csv("keywords.csv", encoding='utf-8', index=False)     
print(df.head(20))
