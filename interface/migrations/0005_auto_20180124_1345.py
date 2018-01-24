# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0004_twitfr'),
    ]

    operations = [
        migrations.AlterField(
            model_name='twitfr',
            name='Polarity',
            field=models.FloatField(default=0.0),
        ),
    ]
