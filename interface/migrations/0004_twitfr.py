# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0003_auto_20180123_1355'),
    ]

    operations = [
        migrations.CreateModel(
            name='TwitFR',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('tweet_text', models.CharField(max_length=280)),
                ('clean_text', models.CharField(max_length=280)),
                ('user', models.CharField(max_length=50)),
                ('nb_retweet', models.IntegerField(default=0)),
                ('nb_like', models.IntegerField(default=0)),
                ('Polarity', models.FloatField()),
                ('Sentiment', models.CharField(max_length=15)),
                ('Date', models.DateTimeField(verbose_name='date published')),
            ],
        ),
    ]
