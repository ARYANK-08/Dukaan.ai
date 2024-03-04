

from django.db import models

class Order(models.Model):
    order_id = models.CharField(max_length=50)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    profit = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.IntegerField()
    category = models.CharField(max_length=100)
    sub_category = models.CharField(max_length=100)
    payment_mode = models.CharField(max_length=50)

    def __str__(self):
        return self.order_id

class Customer(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    order_date = models.DateField()
    customer_name = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    city = models.CharField(max_length=100)

    def __str__(self):
        return self.customer_name
