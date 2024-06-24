from django.db import models

class HealthApplication(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    keywords = models.TextField()

    def __str__(self):
        return self.name
# Create your models here.
