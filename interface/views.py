#!/usr/bin/env python
#-*- coding: utf-8 -*-
from django.shortcuts import render
# Create your views here.

from django.http import HttpResponse

from .models import Twit, TwitFR
import textblob
from textblob import TextBlob
import datetime
from django.utils import timezone
import time
import tweepy 
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from difflib import SequenceMatcher
from twitter import Twitter, OAuth
import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer 
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import sqlite3
import re
import string
from sklearn.naive_bayes import MultinomialNB # For Text classification
from array import array
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance



ACCESS_TOKEN = "286164472-mCiXEbq6DNFH0Xosfw5gfqeGAFDf0yOZEKaH2Aqz"
ACCESS_SECRET = "Zxf07ze7g2aKbokA3zr6GtU8Bs78WsoDm4poTwe9wqkJX"
CONSUMER_KEY = "PtBKawruJYYUEZ4CkPfZi3pU4"
CONSUMER_SECRET = "IRNGOK1XDHlGaqnFHCC8H59z0ltxNUVvWSrDRwCg7qAUA1Iqlr"


nltk.download('stopwords')
tweet_tokenizer = TweetTokenizer() 
punctuation = list(string.punctuation) 
stopword_list = stopwords.words('english') + punctuation + ['rt', 'via', '...', 'AT_USER', 'URL', "'"]


# Dictionnaire pour les tweets en français 

Verbes_Positifs = ["améliorer", "complimenter", "plaire", "sourire", "amuser", "rigoler", "feliciter", "remercier", "adorer", "admirer", "encourager",
                   "encenser", "respecter"]
                   
Verbes_Negatifs = ["engueuler", "menacer", "rompre", "mentir", "facher", "insulter", "agresser", "attaquer", "arnaquer", "duper", "fatiguer", "tromper","pleurer", "dégouter", "boycotter"]

Concepts_Negatifs = ["tristesse", "tragédie", "catastrophe", "maladie", "traitrise", "déception", "médisance", "delation", "hypocrisie", "maléfice", "mensonge", 
                     "malhonnêteté", "foutage", "enervement", "radinerie", "escroquerie", "nullité", "incompétence", "violence", "bétise", "dégout", "agression",
                     "peur","véhémence","pls"]
                     
Concepts_Positifs = ["joie", "plaisir", "bonheur", "bienveillance",  "gentillesse", "honnêteté", "fidélité", "intelligence", "bénéfice", "sympathie"]


Adjectifs_Positifs = ["content","heureux","joyeux", "enthousiaste", "ravi", "optimiste", "bienveillant", "bien", "fort", "performant", "intelligent", "malin", 
                      "énergique", "gentils", "honnête", "intégre", "amical", "cordial", "authentique", "audacieux", "super"]
                      
Adjectifs_Negatifs = ["triste", "malheureux", "déprimé", "désabusé", "pessimiste", "enervé", "malveillant", "mauvais", "faible", "incompétent", "idiot", "bète", 
                      "stupide", "amorphe", "malhonnête", "fatigué", "méchant", "impoli", "agressif", "con", "roublard", "opportuniste"]

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 

    
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


# toutes les fonctions qui suivent concernent l'analyse en français
def similar(a, b):
	return SequenceMatcher(None, a, b).ratio()

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def comparaison_Cpositif(decoupage, j):
        a = 0
        for i in range(0, len(Concepts_Positifs)):     
                if similar(decoupage[j], Concepts_Positifs[i]) > a:       
                        a = similar(decoupage[j], Concepts_Positifs[i])
        print(a)
        return a 
             
def comparaison_Cnegatif(decoupage, j):
        a = 0
        for i in range(0, len(Concepts_Negatifs)):     
                if similar(decoupage[j], Concepts_Negatifs[i]) > a:       
                        a = similar(decoupage[j], Concepts_Negatifs[i])
        print(a)
        return a  
        
def comparaison_APositif(decoupage, j):
        a = 0
        for i in range(0, len(Adjectifs_Positifs)):     
                if similar(decoupage[j], Adjectifs_Positifs[i]) > a:       
                        a = similar(decoupage[j], Adjectifs_Positifs[i])
        print(a)
        return a         
        
def comparaison_ANegatif(decoupage, j):
        a = 0
        for i in range(0, len(Adjectifs_Negatifs)):     
                if similar(decoupage[j], Adjectifs_Negatifs[i]) > a:       
                        a = similar(decoupage[j], Adjectifs_Negatifs[i])
        print(a)
        return a 
          
def comparaison_Vnegatif(decoupage, j):
        a = 0
        for i in range(0, len(Verbes_Negatifs)):     
                if similar(decoupage[j], Verbes_Negatifs[i]) > a:       
                        a = similar(decoupage[j], Verbes_Negatifs[i])
        print(a)
        return a
        
def comparaison_Vpositif(decoupage, j):
        a = 0
        for i in range(0, len(Verbes_Positifs)):     
                if similar(decoupage[j], Verbes_Positifs[i]) > a:       
                        a = similar(decoupage[j], Verbes_Positifs[i])
        print(a)
        return a

def somme_Cpositif(tableau,decoupage):
        a = 0
        for i in range(0, len(decoupage)):     
             a = a + tableau[2][i]
        return a 
             
def somme_Cnegatif(tableau,decoupage):
        a = 0
        for i in range(0, len(decoupage)):     
             a = a + tableau[3][i]
        return a  
        
def somme_Apositif(tableau,decoupage):
        a = 0
        for i in range(0, len(decoupage)):     
             a = a + tableau[4][i]
        return a         
        
def somme_Anegatif(tableau,decoupage):
        a = 0
        for i in range(0, len(decoupage)):     
             a = a + tableau[5][i]
        return a           
    
def somme_Vnegatif(tableau,decoupage):
        a = 0
        for i in range(0, len(decoupage)):     
             a = a + tableau[1][i]
        return a
        
def somme_Vpositif(tableau,decoupage):
        a = 0
        for i in range(0, len(decoupage)):     
             a = a + tableau[0][i]
        return a

def remplissage(tableau, decoupage):
	for j in range(0, len(decoupage)): 
                tableau[0][j] = comparaison_Vpositif(decoupage, j)
                tableau[1][j] = comparaison_Vnegatif(decoupage, j)
                tableau[2][j] = comparaison_Cpositif(decoupage, j)
                tableau[3][j] = comparaison_Cnegatif(decoupage, j)
                tableau[4][j] = comparaison_APositif(decoupage, j)
                tableau[5][j] = comparaison_ANegatif(decoupage, j)
                
	return tableau
	
def Valeur_unique(tableau, decoupage):
        for j in range(0, len(decoupage)):
            max = np.zeros((len(decoupage)), dtype='f')
            for i in range(0,6):
                if tableau[i][j] > max[j]: 
                     max[j] = tableau[i][j]
            for i in range(0,6): 
                     if tableau[i][j] < max[j]:
                        tableau[i][j] = 0
        return tableau              

def Valeur_1(tableau, decoupage):
        for i in range(0,6):
                for j in range(0, len(decoupage)):
                        if tableau[i][j] > 0: tableau[i][j] = 1
        return tableau                      
             	
def seuil(tableau, decoupage):    
         for i in range(0, 6):
                for j in range(0, len(decoupage)):
                        if tableau[i][j] < 0.75:
                                tableau[i][j] = 0
         return tableau


# Fonction appelée pour la page d'accueil
def home(request):

    return render(request, 'interface/accueil.html', locals())

# Fonction appelée pour la page analyse Anglais
def analyse(request):
   
    return render(request, 'interface/analyse.html', locals())
  
# Fonction d'extraction des tweets vers la base de donnée sqlite
def recherche(request):
     Twit.objects.all().delete()
     if request.method == 'GET':
      hashtag = request.GET.get('search_box', None)
      nombre = request.GET.get('number', None)
     
     auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
     auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
     # Return API with authentication:
     api = tweepy.API(auth)
     nombre = int(nombre)
     nombre2 = nombre
     pos = 0.0
     neg = 0.0
     neu = 0.0

# On identifie le tweet qui a l'id maximum
     tweets = api.search(q = hashtag, lang = 'en', count=1)
     max = tweets[0].id
     while (nombre > 100):
      tweets = api.search(q = hashtag,lang = 'en', count=100, max_id = max)
      nombre = nombre - 100
      max = tweets[99].id
     
# Grace à 2 boucles for on peut extraire pile le nombre de tweets rentrés     
      for tweet in tweets[:100]:
       clean_text = clean_tweet(tweet.text)
       twit = TextBlob(tweet.text)
       if twit.polarity == 0:
        sentiment = "neutre"
        neu = neu + 1
       if twit.polarity > 0:
        sentiment = "positif"
        pos = pos + 1
       if twit.polarity < 0:
        sentiment = "negatif"
        neg = neg + 1
       q = Twit(tweet_text=tweet.text, clean_text = clean_text, user = tweet.user , nb_retweet = tweet.retweet_count , 
       nb_like = tweet.favorite_count, Sentiment = sentiment, Polarity = twit.polarity, Date=tweet.created_at)
       q.save()
     
  
     tweets = api.search(q = hashtag,lang = 'en', count=nombre,  max_id = max)
     for tweet in tweets[:nombre]:
      clean_text = clean_tweet(tweet.text)
      twit = TextBlob(tweet.text)
      if twit.polarity == 0:
       sentiment = "neutre"
       neu = neu + 1
      if twit.polarity > 0:
       sentiment = "positif"
       pos = pos + 1
      if twit.polarity < 0:
       sentiment = "negatif"
       neg = neg + 1
      q = Twit(tweet_text=tweet.text, clean_text = clean_text, user = tweet.user , nb_retweet = tweet.retweet_count , 
      nb_like = tweet.favorite_count, Sentiment = sentiment, Polarity = twit.polarity, Date=tweet.created_at)
      q.save()
     neu = (neu*100)
     neg = (neg*100)
     pos = (pos*100)
     neg = neg/nombre2
     pos = pos/nombre2
     neu = neu/nombre2
     return render(request, "interface/resultatEN.html", {"neg": neg, "pos": pos, "neutre": neu, "nombre": nombre2, "hashtag": hashtag })



# Fonction de nettoyage des tweets

def clean_tweet(tweet):
    tweet = tweet.lower() # Miniscule
    tweet = re.sub('[\s]+', ' ', tweet) # Suppresion des espaces en trop
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', tweet) # Suppresion des liens externes
    tweet = re.sub('@[^\s]+','AT_USER', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # Suppresion des Hashtags
    tweet = tweet.strip('\'"')
    return tweet

def replace_similar_characters(s): # For Example finalllllly becomes finally
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# Fonction Feature dans le cadre des algorithmes ML
def get_feature_vector(tweet):
        feature_vector = []
        stopword_list = stopwords.words('english') + punctuation + ['rt', 'via', '...', 'AT_USER', 'URL', 'https', 'https',"'"]
        words = list(normalization(tokenization(tweet, tokenizer=tweet_tokenizer, stopwords= stopword_list), stopwords= stopword_list))
        for word in words:
            word = replace_similar_characters(word)  
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word) # Check if the word starts with a letter or a number
            if val is None:
                continue
            else:    
                feature_vector.append(word.lower())
        return feature_vector

def tokenization(tweet, tokenizer = TweetTokenizer(), stopwords = []):
    tokens = tokenizer.tokenize(tweet)
    return [tok for tok in tokens if tok not in stopwords and not tok.isdigit()]


def words_in_tweets(labeled_tweets):
    list_of_words = []
    for (words, sentiment) in labeled_tweets:
        list_of_words.extend(words)
    return list_of_words

def join_words(words):
    return( ' '.join(item for item in words))   
         
def normalization(tokens, stopwords = []):
    token_map = {
        "i'm": "i am",
        "you're": "you are", 
        "it's": "it is", 
        "we're": "we are", 
        "we'll": "we will",
        "ain't": "are not",
        "ive" : "i have",
    }
    
    for token in tokens:
        if token in token_map.keys():
            for item in token_map[token].split():
                if item not in stopwords:
                    yield item
        else:
            yield token



# Partie sur les tweets en Français 

def analysefr(request):

    return render(request, 'interface/analysefr.html', locals())

# Sentiment des tweets français
def GetSentiment(clean_text): 
    
     decoupage = preprocess(clean_text)
     tableau_coefficient = np.zeros((7, len(decoupage)), dtype='f')
     tableau = remplissage(tableau_coefficient, decoupage)
     tableau = seuil(tableau, decoupage)
     tableau = Valeur_unique(tableau, decoupage)
     tableau = Valeur_1(tableau, decoupage)
     Vpositif = somme_Vpositif(tableau,decoupage)
     Vnegatif = somme_Vnegatif(tableau,decoupage)
     Apositif = somme_Apositif(tableau,decoupage)
     Anegatif = somme_Anegatif(tableau,decoupage)
     Cpositif = somme_Cpositif(tableau,decoupage)
     Cnegatif = somme_Cnegatif(tableau,decoupage)
     analyse = 0
     if (Apositif > 0): 
       analyse = 1
     if (Cpositif > 0): 
        analyse = 1
     if (Vpositif > 0): 
        analyse = 1
         
     if (Anegatif > 0): 
       analyse = - 1 
     if (Vnegatif > 0): 
       analyse = - 1 
     if (Cnegatif > 0): 
       analyse = - 1
     
     return analyse

def recherchefr(request):


    TwitFR.objects.all().delete()
    if request.method == 'GET':
     hashtag = request.GET.get('search_box', None)
     nombre = request.GET.get('number', None)
     
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    # Return API with authentication:
    api = tweepy.API(auth)
    nombre = int(nombre)
    nombre2 = nombre

# On identifie le tweet qui a l'id maximum
    tweets = api.search(q = hashtag, lang = 'fr', count=1)
    max = tweets[0].id
    while (nombre > 100):
     tweets = api.search(q = hashtag,lang = 'fr', count=100, max_id = max)
     nombre = nombre - 100
     max = tweets[99].id
     
# Grace à 2 boucles for on peut extraire pile le nombre de tweets rentrés     
     for tweet in tweets[:100]:
      clean_text = clean_tweet(tweet.text)
      analyse = GetSentiment(clean_text)
      if analyse == 1:
        Sentiment = "Positif"
      if analyse == 0: 
        Sentiment = "Neutre"
      if analyse == - 1:
        Sentiment = "Negatif"
      q = TwitFR(tweet_text=tweet.text, clean_text = clean_text, user = tweet.user , nb_retweet = tweet.retweet_count , 
      nb_like = tweet.favorite_count, Date=tweet.created_at, Polarity = analyse, Sentiment = Sentiment)
      q.save()
     
  
    tweets = api.search(q = hashtag,lang = 'fr', count=nombre,  max_id = max)
    for tweet in tweets[:nombre]:
     clean_text = clean_tweet(tweet.text)
     analyse = GetSentiment(clean_text)
     
     if analyse == 1:
        Sentiment = "Positif"
     if analyse == 0: 
        Sentiment = "Neutre"
     if analyse == - 1:
        Sentiment = "Negatif"  
     q = TwitFR(tweet_text=tweet.text, clean_text = clean_text, user = tweet.user , nb_retweet = tweet.retweet_count , 
     nb_like = tweet.favorite_count, Date=tweet.created_at, Polarity = analyse, Sentiment = Sentiment)
     q.save()
     
    nombre = nombre2
    conn = sqlite3.connect("db.sqlite3")
    cur = conn.cursor()
    cur.execute("select Sentiment from interface_twitfr")
    results = cur.fetchall()
    Nb_Positif = 0.0
    Nb_Negatif = 0.0
    Nb_Neutre = 0.0
    for i in range(0, nombre):
     variable = str(results[i])
     b = similar(variable, "Positif")
     if b > 0.5:
        Nb_Positif = Nb_Positif + 1
  
    
    Nb_Positif = 100*Nb_Positif
    Nb_Positif = Nb_Positif/nombre2

  
    for i in range(0, nombre):
     variable = str(results[i])
     a = similar(variable, "Negatif")
    
     if a > 0.5:
        Nb_Negatif = Nb_Negatif + 1
    
   
    Nb_Negatif = 100*Nb_Negatif
    Nb_Negatif = Nb_Negatif/nombre2
    

    for i in range(0, nombre):
     variable = str(results[i])
     c = similar(variable, "Neutre")

     if c > 0.5:
       Nb_Neutre  = Nb_Neutre + 1

    Nb_Neutre = 100*Nb_Neutre
    Nb_Neutre = Nb_Neutre/nombre2




    return render(request, "interface/resultat.html", {"neg": Nb_Negatif, "pos": Nb_Positif, "neutre": Nb_Neutre, "nombre": nombre2, "hashtag": hashtag })

   

