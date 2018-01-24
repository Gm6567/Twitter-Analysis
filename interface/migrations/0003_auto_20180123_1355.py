# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0002_remove_twit_nb_answer'),
    ]

    operations = [
        migrations.AlterField(
            model_name='twit',
            name='Polarity',
            field=models.FloatField(),
        ),
    ]
