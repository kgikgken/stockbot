import requests
import os

def send_line(message):
    token = os.getenv("LINE_TOKEN")
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "to": "YOUR_USER_ID",  # ←あとで書き換える
        "messages": [
            {"type": "text", "text": message}
        ]
    }
    requests.post(url, headers=headers, json=data)

def main():
    send_line("テストメッセージ：GitHub Actions から送信成功！")

if __name__ == "__main__":
    main()
