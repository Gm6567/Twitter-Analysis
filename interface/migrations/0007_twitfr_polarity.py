# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0006_auto_20180124_1353'),
    ]

    operations = [
        migrations.AddField(
            model_name='twitfr',
            name='Polarity',
            field=models.FloatField(default=0),
        ),
    ]
