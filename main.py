import requests
from flask import Flask, render_template, request
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta, timezone
import requests  
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import numpy as np
import requests
import yfinance
import re

from newspaper import Article
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.tokenize import word_tokenize

app = Flask(__name__)
model = None
tokenizer = None
X_padded = None
label_encoder = None

def nlp_process(text):
    text = remove_urls(text)
    text = remove_html(text)
    text = lower(text)
    text = remove_num(text)
    text = punct_remove(text)
    text = remove_stopwords(text)
    text = remove_mention(text)
    text = remove_hash(text)
    text = remove_space(text)
    return text

def train_model():
    global model
    global model, tokenizer, X_padded, label_encoder
    finance = pd.read_csv('./training_model/datasets/finance.csv')
    finance = finance.rename(columns={"headline":"headline_en", "translated_text": "headline_ru"})
    gpt_train = pd.read_csv('./training_model/datasets/gpt_train.csv')
    gpt_train = gpt_train.rename(columns={"headline":"headline_en", "translated_text": "headline_ru"})
    risks = pd.read_csv('./training_model/datasets/risks.csv')
    risks = risks.rename(columns={"headline":"headline_en", "translated_text": "headline_ru"})
    user_responses = pd.read_csv('./training_model/datasets/user_respsonses.csv')
    user_responses = user_responses.rename(columns={"headline":"headline_en", "translated_text": "headline_ru"})
    df = pd.concat([finance, gpt_train, user_responses, risks], ignore_index=True)

    def nlp_process(df):
        df['headline_en']=df['headline_en'].apply(lambda x:remove_urls(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_html(x))
        df['headline_en']=df['headline_en'].apply(lambda x:lower(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_num(x))
        df['headline_en']=df['headline_en'].apply(lambda x:punct_remove(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_stopwords(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_mention(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_hash(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_space(x))
        return df
    
    df = nlp_process(df)
    ru_df = df[['headline_ru', 'sentiment']]
    en_df = df[['headline_en', 'sentiment']]
    df = pd.DataFrame({"headline": pd.concat([en_df['headline_en'], ru_df['headline_ru']], ignore_index=True), "sentiment": pd.concat([en_df['sentiment'], ru_df['sentiment']], ignore_index=True)})
    df = df.iloc[np.random.permutation(df.index)].reset_index(drop=True)
    X = df['headline']
    y = df['sentiment']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=X_padded.shape[1]))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax')) 

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

    loss, accuracy = model.evaluate(X_test, y_test)
    model.save('sentiment_model_v5.keras')

    tokenizer = tokenizer
    X_padded = X_padded
    label_encoder = label_encoder

    return model
", ".join(stopwords.words('russian'))
STOPWORDS = set(stopwords.words('russian'))

def remove_urls(text):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', text)

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def lower(text):
    low_text = text.lower()
    return low_text

def remove_num(text):
    remove = re.sub(r'\d+', '', text)
    return remove

def punct_remove(text):
    punct = re.sub(r"[^\w\s\d]", "", text)
    return punct

def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_mention(x):
    text = re.sub(r'@\w+', '', x)
    return text

def remove_hash(x):
    text = re.sub(r'#\w+', '', x)
    return text

def remove_space(text):
    space_remove = re.sub(r"\s+", " ", text).strip()
    return space_remove


 
# ************* API KEYS *************
COIN_MARKET_API_KEY = "cde45351-3b00-49ff-8b0a-bc62c8e88295"
# ************************************


@app.route('/')
def main():
    global model
    if model is None:
        model = train_model()
    current_currencies = getCryptoData()
    return render_template('main.html', currencies = current_currencies)

@app.route('/details/<id>')
def details(id):
    interval = '1m' 
    graph_html = create_graph(id, interval)
    news_data = get_news_description(id)
    news_latest = latest_news()
    print(news_latest)
    return render_template('details.html', graph_html=graph_html, id=id, latest_data = news_latest)


# ************* Functions for main page *************
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
            'symbol': currency['symbol'],
            'price': currency['quote']['USD']['price'],
            'volume_24h': currency['quote']['USD']['volume_24h'],
            'market_cap': currency['quote']['USD']['market_cap']
        }
        crypto_array.append(currency_info)
    return crypto_array


def formatLargeNumber(value):
    if value is None:
        return "N/A"
    
    value = float(value)

    magnitude = ""
    if abs(value) >= 1.0e9:
        value /= 1.0e9
        magnitude = "B"
    elif abs(value) >= 1.0e6:
        value /= 1.0e6
        magnitude = "M"

    return "{:.2f}{}".format(value, magnitude)
# ********************************************

# ************* Functions for details page *************

def get_crypto_data(id, interval):
    end_date = datetime.now()
    
    if interval == '1d':
        start_date = end_date - timedelta(days=1)
        crypto_currencies = yf.download(f'{id}-USD', interval='1m', start=start_date, end=end_date)
    elif interval == '1w':
        start_date = end_date - timedelta(weeks=1)
        crypto_currencies = yf.download(f'{id}-USD', interval='1d', start=start_date, end=end_date)
    else:
        crypto_currencies = yf.download(f'{id}-USD', interval='1m', period='1d')
    
    return crypto_currencies

def create_graph(id, interval):
    crypto_currencies = get_crypto_data(id, interval)
    news_time, news_links = get_news_data(id)

    fig = px.line(crypto_currencies, x=crypto_currencies.index, y='Close', title=f'{id} Price Over Time')
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Price (USD)')
    fig.update_layout(showlegend=False)

    for time, link in zip(news_time, news_links):
        time = datetime.fromisoformat(time)
        closest_time_idx = np.abs(crypto_currencies.index - time).argmin()

        fig.add_trace(
            go.Scatter(
                x=[time],
                y=[crypto_currencies.loc[crypto_currencies.index[closest_time_idx], 'Close']],
                mode='markers',
                marker=dict(size=10),
                text=f"<a href='{link}' target='_blank'>News Link</a>",
                name='News Point',
            )
        )

    graph_html = fig.to_html(full_html=False, config={'displayModeBar': False, 'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']})

    graph_html += """
        <script>
            // Resize the plot when the window size changes
            window.addEventListener('resize', function() {
                Plotly.relayout('your-plot-div-id', {
                    width: window.innerWidth * 0.9,  // Adjust the multiplier as needed
                    height: window.innerHeight * 0.7  // Adjust the multiplier as needed
                });
            });
        </script>
        """

    return graph_html

def get_news_data(id):
    news_time = []
    news_links = []

    date_format = "%Y-%m-%d %H:%M:%S%z"

    url_en = f'https://api.blockchair.com/news?q=language(en),title(~{id}),or,description(~{id})'
    en_news_data = requests.get(url_en).json()

    for i in range(len(en_news_data['data'])):
        news_time.append(str(datetime.strptime(en_news_data['data'][i]['time'][:-3] + ':00+0000', date_format)))
        news_links.append(en_news_data['data'][i]['link'])

    url_ru = f'https://api.blockchair.com/news?q=language(ru),title(~{id}),or,description(~{id})'
    ru_news_data = requests.get(url_ru).json()

    for i in range(len(ru_news_data['data'])):
        news_time.append(str(datetime.strptime(ru_news_data['data'][i]['time'][:-3] + ':00+0000', date_format)))
        news_links.append(ru_news_data['data'][i]['link'])

    return news_time, news_links

def get_news_description(id):
    news_data = []

    date_format = "%Y-%m-%d %H:%M:%S%z"

    url_en = f'https://api.blockchair.com/news?q=language(en),title(~{id}),or,description(~{id})'
    en_news_data = requests.get(url_en).json()

    for i in range(len(en_news_data['data'])):
        news_time = datetime.strptime(en_news_data['data'][i]['time'][:-3] + ':00+0000', date_format).replace(tzinfo=timezone.utc)

        if datetime.now(timezone.utc) - news_time < timedelta(minutes=10):
            url = en_news_data['data'][i]['link']
            sentiment = analyze_sentiment(en_news_data['data'][i]['description'])
            news_data.append({'url': url, 'time': news_time, 'sentiment': sentiment})


    url_ru = f'https://api.blockchair.com/news?q=language(ru),title(~{id}),or,description(~{id})'
    ru_news_data = requests.get(url_ru).json()

    for i in range(len(ru_news_data['data'])):
        news_time = datetime.strptime(ru_news_data['data'][i]['time'][:-3] + ':00+0000', date_format).replace(tzinfo=timezone.utc)

        if datetime.now(timezone.utc) - news_time < timedelta(minutes=10):
            url = ru_news_data['data'][i]['link']
            sentiment = analyze_sentiment(ru_news_data['data'][i]['description'])
            news_data.append({'url': url, 'time': news_time, 'sentiment': sentiment})

    return news_data

def analyze_sentiment(description):
    description = nlp_process(pd.DataFrame({"headline_en": [description]}))['headline_en'][0]
    description_sequence = tokenizer.texts_to_sequences([description])
    description_padded = pad_sequences(description_sequence, maxlen=X_padded.shape[1])
    sentiment_probabilities = model.predict(description_padded)[0]
    predicted_sentiment_index = np.argmax(sentiment_probabilities)
    predicted_sentiment = label_encoder.classes_[predicted_sentiment_index]

    return predicted_sentiment


def latest_news():


    news_time_en = []
    news_links_en = []

    news_time_ru = []
    news_links_ru = []

    date_format = "%Y-%m-%d %H:%M:%S%z"

    def getNews():
        url_en = "https://api.blockchair.com/news?q=language(en),title(~btc),or,title(~Bitcoin),or,description(~Bitcoin)"
        try:
            response = requests.get(url_en)
            if response.status_code == 200:
                news_data = response.json()
                for i in range(len(news_data['data'])):
                    news_time_en.append(str(datetime.strptime(news_data['data'][i]['time'][:-3] + ':00+0000', date_format)))
                    news_links_en.append(news_data['data'][i]['link'])
            else:
                print(f"Error: {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")
        url_ru = "https://api.blockchair.com/news?q=language(ru),title(~btc),or,title(~Bitcoin),or,description(~Bitcoin)"
        try:
            response = requests.get(url_ru)
            if response.status_code == 200:
                news_data = response.json()
                for i in range(len(news_data['data'])):
                    news_time_ru.append(str(datetime.strptime(news_data['data'][i]['time'][:-3] + ':00+0000', date_format)))
                    news_links_ru.append(news_data['data'][i]['link'])
            else:
                print(f"Error: {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")

    getNews()

    print(news_time_ru)
    print(news_links_ru)

    print(news_time_en)
    print(news_links_en)

    news_description_en = []
    errors_en = []

    news_description_ru = []
    errors_ru = []

    def getNewsDescription():
        for i in range(len(news_links_en)):
            try:
                article = Article(news_links_en[i], language="en")
                article.download()
                article.parse()
                article.nlp()
                news_description_en.append(article.summary)
            except Exception as e:
                print(f"Error processing news link {news_links_en[i]}: {str(e)}")
                errors_en.append(i)
        
        for i in range(len(news_links_ru)):
            try:
                article = Article(news_links_ru[i], language="ru")
                article.download()
                article.parse()
                article.nlp()
                news_description_ru.append(article.summary)
            except Exception as e:
                print(f"Error processing news link {news_links_ru[i]}: {str(e)}")
                errors_ru.append(i)

    getNewsDescription()

    print(news_description_en)
    print(errors_en)

    print(news_description_ru)
    print(errors_ru)

    if len(news_time_en) != len(news_description_en):
        for error in errors_en:
            news_time_en.remove(news_time_en[error])

    print(len(news_time_en))
    print(len(news_description_en))

    if len(news_time_ru) != len(news_description_ru):
        for error in errors_ru:
            news_time_ru.remove(news_time_ru[error])

    print(len(news_time_ru))
    print(len(news_description_ru))

    try:
        crypto_currencies = yfinance.download('BTC-USD', interval='1m')
    except Exception as e:
        print(f"Error processing")

    print(crypto_currencies)
    price_diff_en = []
    price_diff_ru = []

    for i in range(len(news_description_en)):
        fTargetRow = crypto_currencies[crypto_currencies['Close'].index >= news_time_en[i]].head(1)
        sTargetRow = crypto_currencies[crypto_currencies['Close'].index >= news_time_en[i]].head(11).tail(1)
        fPrice = fTargetRow['Close'].values
        sPrice = sTargetRow['Close'].values

        price_diff_en.append(fPrice[0] - sPrice[0])

    print(price_diff_en)

    for i in range(len(news_description_ru)):
        fTargetRow = crypto_currencies[crypto_currencies['Close'].index >= news_time_ru[i]].head(1)
        sTargetRow = crypto_currencies[crypto_currencies['Close'].index >= news_time_ru[i]].head(11).tail(1)
        fPrice = fTargetRow['Close'].values
        sPrice = sTargetRow['Close'].values

        price_diff_ru.append(fPrice[0] - sPrice[0])

    print(price_diff_ru)

    ", ".join(stopwords.words('english'))
    STOPWORDS = set(stopwords.words('english'))

    def remove_urls(text):
        url_remove = re.compile(r'https?://\S+|www\.\S+')
        return url_remove.sub(r'', text)

    def remove_html(text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)

    def lower(text):
        low_text= text.lower()
        return low_text

    def remove_num(text):
        remove= re.sub(r'\d+', '', text)
        return remove

    def punct_remove(text):
        punct = re.sub(r"[^\w\s\d]","", text)
        return punct

    def remove_stopwords(text):
        """custom function to remove the stopwords"""
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

    def remove_mention(x):
        text=re.sub(r'@\w+','',x)
        return text

    def remove_hash(x):
        text=re.sub(r'#\w+','',x)
        return text

    def remove_space(text):
        space_remove = re.sub(r"\s+"," ",text).strip()
        return space_remove
    
    def nlp_process_en(df):
        df['headline_en']=df['headline_en'].apply(lambda x:remove_urls(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_html(x))
        df['headline_en']=df['headline_en'].apply(lambda x:lower(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_num(x))
        df['headline_en']=df['headline_en'].apply(lambda x:punct_remove(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_stopwords(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_mention(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_hash(x))
        df['headline_en']=df['headline_en'].apply(lambda x:remove_space(x))
        return df

    current_news_en = pd.DataFrame({"headline_en": news_description_en, "diff_price_en":price_diff_en})

    current_news_en = current_news_en[current_news_en['headline_en'] != ""]

    current_news_en = nlp_process_en(current_news_en)

    current_news_ru = pd.DataFrame({"headline_ru": news_description_ru, "diff_price_ru":price_diff_ru})

    current_news_ru = current_news_ru[current_news_ru['headline_ru'] != ""]

    ", ".join(stopwords.words('russian'))
    STOPWORDS = set(stopwords.words('russian'))

    def remove_urls_ru(text):
        url_remove = re.compile(r'https?://\S+|www\.\S+')
        return url_remove.sub(r'', text)

    def remove_html_ru(text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)

    def lower_ru(text):
        low_text= text.lower()
        return low_text

    def remove_num_ru(text):
        remove= re.sub(r'\d+', '', text)
        return remove

    def punct_remove_ru(text):
        punct = re.sub(r"[^\w\s\d]","", text)
        return punct

    def remove_stopwords_ru(text):
        """custom function to remove the stopwords"""
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

    def remove_mention_ru(x):
        text=re.sub(r'@\w+','',x)
        return text

    def remove_hash_ru(x):
        text=re.sub(r'#\w+','',x)
        return text

    def remove_space_ru(text):
        space_remove = re.sub(r"\s+"," ",text).strip()
        return space_remove

    def nlp_process_ru(df):
        df['headline_ru']=df['headline_ru'].apply(lambda x:remove_urls_ru(x))
        df['headline_ru']=df['headline_ru'].apply(lambda x:remove_html_ru(x))
        df['headline_ru']=df['headline_ru'].apply(lambda x:lower_ru(x))
        df['headline_ru']=df['headline_ru'].apply(lambda x:remove_num_ru(x))
        df['headline_ru']=df['headline_ru'].apply(lambda x:punct_remove_ru(x))
        df['headline_ru']=df['headline_ru'].apply(lambda x:remove_stopwords_ru(x))
        df['headline_ru']=df['headline_ru'].apply(lambda x:remove_mention_ru(x))
        df['headline_ru']=df['headline_ru'].apply(lambda x:remove_hash_ru(x))
        df['headline_ru']=df['headline_ru'].apply(lambda x:remove_space_ru(x))
        return df
    

    current_news_ru = nlp_process_ru(current_news_ru)

    ru_current_df = current_news_ru[['headline_ru', 'diff_price_ru']]

    en_current_df = current_news_en[['headline_en', 'diff_price_en']]
    current_news = pd.DataFrame({"headline": pd.concat([ru_current_df['headline_ru'], en_current_df['headline_en']], ignore_index=True),
                             "diff_price": pd.concat([ru_current_df['diff_price_ru'], en_current_df['diff_price_en']], ignore_index=True)})
    X_new = current_news['headline']

    X_new_sequences = tokenizer.texts_to_sequences(X_new)
    X_new_padded = pad_sequences(X_new_sequences, maxlen=X_padded.shape[1])

    saved_model_path = 'sentiment_model_v5.keras'
    model = load_model(saved_model_path)

    predictions = model.predict(X_new_padded)

    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    current_news['predicted_sentiment'] = predicted_labels

    def analyze_news_effect(predicted_sentiment, diff_price):
        if diff_price > 0 and predicted_sentiment == 'positive':
            return "The news affected the price: positive effect"
        elif diff_price < 0 and predicted_sentiment == 'negative':
            return "The news affected the price: negative effect"
        elif -5 <= diff_price <= 5 and predicted_sentiment == 'neutral':
            return "The news did not affect the price: neutral effect"
        elif diff_price < 0 and predicted_sentiment == 'positive':
            return "The news did not affect the price: negative effect"
        elif diff_price > 0 and predicted_sentiment == 'negative':
            return "The news did not affect the price: positive effect"
        elif predicted_sentiment == 'neutral' and diff_price > 0:
            return "The news did not affect the price: positive effect"
        elif predicted_sentiment == 'neutral' and diff_price < 0:
            return "The news did not affect the price: negative effect"
        
    current_news['relationship'] = current_news.apply(lambda row: analyze_news_effect(row['predicted_sentiment'], row['diff_price']), axis=1)
    return current_news.to_dict(orient='records')
    



app.jinja_env.filters['formatLargeNumber'] = formatLargeNumber


if __name__ == '__main__':
    app.run(debug=True)