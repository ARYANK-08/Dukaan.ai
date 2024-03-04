from django.test import TestCase

# Create your tests here.
import os
from twilio.rest import Client


def send_report_via_sms():
    account_sid = 'ACbf6cee76099df8fffbf3b1956c747fcd'
    auth_token = '727f773bf34138354ce81640b466ce32'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
                                    body=f'Hello, Aryan \nProduct Name : Bisleri Bottle is missing \nShelf Location : D5 shelf \nPlease refill and find the real-time video below :',

    to='whatsapp:+919653484071'
    )

    print(message.sid)

send_report_via_sms()