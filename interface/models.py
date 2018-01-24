from __future__ import unicode_literals

from django.db import models


class Twit(models.Model):
    tweet_text = models.CharField(max_length=280)
    clean_text = models.CharField(max_length=280)
    user = models.CharField(max_length=50)
    nb_retweet = models.IntegerField(default=0)
    nb_like = models.IntegerField(default=0)
    Polarity = models.FloatField()
    Sentiment = models.CharField(max_length=15) 
    Classification = models.IntegerField(default=0)
    Date = models.DateTimeField('date published')
    
class TwitFR(models.Model):
    tweet_text = models.CharField(max_length=280)
    clean_text = models.CharField(max_length=280)
    user = models.CharField(max_length=50)
    nb_retweet = models.IntegerField(default=0)
    nb_like = models.IntegerField(default=0)
    Polarity = models.FloatField(default=0)
    Sentiment = models.CharField(max_length=15)
    Date = models.DateTimeField('date published')

