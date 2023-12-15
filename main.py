from flask import Flask, render_template
import requests

app = Flask(__name__)

# ************* API KEYS *************
COIN_MARKET_API_KEY = "cde45351-3b00-49ff-8b0a-bc62c8e88295"
# ************************************


@app.route('/')
def main():
    current_currencies = getCryptoData()
    return render_template('main.html', currencies = current_currencies)


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


def formatLargeNumber(value):
    if value is None:
        return "N/A"

    # Convert to a float to ensure that it's a numerical value
    value = float(value)

    # Determine the magnitude of the number
    magnitude = ""
    if abs(value) >= 1.0e9:
        value /= 1.0e9
        magnitude = "B"
    elif abs(value) >= 1.0e6:
        value /= 1.0e6
        magnitude = "M"

    # Format the number with 2 decimal places
    return "{:.2f}{}".format(value, magnitude)
# ********************************************


app.jinja_env.filters['formatLargeNumber'] = formatLargeNumber


if __name__ == '__main__':
    app.run(debug=True)