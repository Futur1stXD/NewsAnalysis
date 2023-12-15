from flask import Flask
import requests

app = Flask(__name__)

# ************* API KEYS *************
COIN_MARKET_API_KEY = "cde45351-3b00-49ff-8b0a-bc62c8e88295"
# ************************************


@app.route('/')
def main():
    current_currencies = getCryptoData()
    print(current_currencies[1])
    return 'Hello World!'


# ************* Useful functions *************
def getCryptoData():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start': '1',
        'limit': '10'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': 'cde45351-3b00-49ff-8b0a-bc62c8e88295',
    }
    result = requests.get(url, params=parameters, headers=headers).json()
    crypto_array = []
    for currency in result['data']:
        currency_info = {
            'name': currency['name'],
            'price': currency['quote']['USD']['price'],
            'volume_24h': currency['quote']['USD']['volume_24h'],
            'market_cap': currency['quote']['USD']['market_cap']
        }
        crypto_array.append(currency_info)
    return crypto_array
# ********************************************


if __name__ == '__main__':
    app.run(debug=True, port=3000)