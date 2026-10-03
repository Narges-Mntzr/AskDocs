from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("knowledge", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="document",
            name="source_name",
        ),
    ]
