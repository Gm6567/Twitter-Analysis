# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0007_twitfr_polarity'),
    ]

    operations = [
        migrations.AddField(
            model_name='twitfr',
            name='Sentiment',
            field=models.CharField(default='neutre', max_length=15),
            preserve_default=False,
        ),
    ]
