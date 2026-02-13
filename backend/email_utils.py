# email_utils.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")  # sender email
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # app password
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))


def send_email(to_email: str, subject: str, body: str):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)


def send_reset_email(to_email: str, otp: str):
    subject = "Your Password Reset OTP"
    body = f"""
Hello,

Your OTP for password reset is: {otp}

If you did not request this, please ignore this email.
"""
    send_email(to_email, subject, body)
