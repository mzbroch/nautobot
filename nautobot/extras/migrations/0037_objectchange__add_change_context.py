# Generated by Django 3.2.13 on 2022-07-08 19:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("extras", "0036_configcontext_locations"),
    ]

    operations = [
        migrations.AddField(
            model_name="objectchange",
            name="change_context",
            field=models.CharField(blank=True, max_length=50),
        ),
        migrations.AddField(
            model_name="objectchange",
            name="change_context_detail",
            field=models.CharField(blank=True, max_length=100),
        ),
    ]
