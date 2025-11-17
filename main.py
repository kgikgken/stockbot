import requests
import os

def send_line(message):
    token = os.getenv("LINE_TOKEN")
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "messages": [
            {"type": "text", "text": message}
        ]
    }
    requests.post(url, headers=headers, json=data)

send_line("GitHub Actions からのテスト送信です！")

