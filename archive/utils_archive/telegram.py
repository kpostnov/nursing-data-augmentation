import math
import requests
from urllib.parse import quote


def send_telegram(message: str) -> None:
    packages_remaining = [message]
    max_messages_num = 40
    while len(packages_remaining) > 0 and max_messages_num > 0:
        curr_package = packages_remaining.pop(0)
        message_sent = telegram_fetch(curr_package)
        if message_sent:
            max_messages_num -= 1
        if not message_sent:
            if len(curr_package) < 10:
                telegram_fetch("Telegram failed")
                break
            num_of_chars_first = math.ceil(len(curr_package) / 2)
            first_package = curr_package[0:num_of_chars_first]
            second_package = curr_package[num_of_chars_first : len(curr_package)]

            packages_remaining.insert(0, second_package)
            packages_remaining.insert(0, first_package)
    if max_messages_num == 0:
        telegram_fetch("Sending failed. Too many messages sent.")


def telegram_fetch(message: str) -> bool:
    message = str(message)

    # TradingBot
    chatId = "-777608805"
    botKey = "1020600091:AAE7JsJ9pnIkhqdjP_sd3_x5uITi8q9rHrU"

    # updates and chatId https://api.telegram.org/bot<YourBOTToken>/getUpdates
    # For \n use %0A message = message.replace(/\n/g, "%0A")
    url = (
        "https://api.telegram.org/bot"
        + botKey
        + "/sendMessage?chat_id="
        + chatId
        + "&text="
        + quote(message)
    )

    try:
        response = (requests.get(url)).json()
        return response["ok"]
    except:
        return False
