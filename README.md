# Astana IT University: Blockchain Hack 2023 

## Project name

News Analysis

## Selected problem

An analyzer of the impact of crypto market news: Correlation of news sentiment with the movement of token prices.

## Team name

Route66

## Participants

* Full name: Мырзаханов Абылайхан Уланулы. Email: 212257@astanait.edu.kz
* Full name: Туран Мирас Серикханулы. Email: 212336@astanait.edu.kz
* Full name: Тулешев Туран Бержанович. Email: 212470@astanait.edu.kz

## Abstract

The goal: to create an NLP model that would qualitatively analyze the text. Create a tool that would help the user understand what news has affected the market.

We have created an NLP model that qualitatively analyzes the text in Russian and English. Its accuracy is ~80-83%. Our project also has functions for analyzing how the news affected the market and by how much.
<img width="939" alt="Снимок экрана 2023-12-20 в 22 54 23" src="https://github.com/Futur1stXD/NewsAnalysis/assets/126179639/4cdb2c2f-8394-4a44-8b9d-59757c17f480">

For interactive purposes, we have created a website to demonstrate our implementation.
![2023-12-21 10 58 46](https://github.com/Futur1stXD/NewsAnalysis/assets/126179639/93312d93-1ab3-42c1-bbed-09a3916ca08a)
<img width="1512" alt="Снимок экрана 2023-12-21 в 10 59 11" src="https://github.com/Futur1stXD/NewsAnalysis/assets/126179639/8cc6b1d9-bd73-4155-8d72-391d7ab52941">
<img width="1512" alt="Снимок экрана 2023-12-21 в 10 59 22" src="https://github.com/Futur1stXD/NewsAnalysis/assets/126179639/059c3912-86fa-4813-bcc3-eefbb52274cb">
<img width="1512" alt="Снимок экрана 2023-12-21 в 10 59 34" src="https://github.com/Futur1stXD/NewsAnalysis/assets/126179639/2471ccd1-24c4-4ed6-9708-4f014c4cabc5">


## Demo video

[Link to a demo video showcasing your project, if any. Ensure it is less than 3 minutes long.]

## How to run

### Prerequisites:

pip3 install pandas
pip3 install newspaper3k
pip3 install yfinance
pip3 install numpy
pip3 install requests
pip3 install nltk
pip3 install scikit-learn
pip3 install tenserflow

For checking our interactive you need install:
pip3 install flask

### Running

[Provide specific commands and environment for building and running your project, preferably in a containerized environment.]

To run the application, ensure that you have the required dependencies installed. You can then execute the script, and the Flask application will start running locally on port 5000 (default).

## Inspirations

[Explain the inspirations behind your project. What motivated your team to work on this idea?]

We took this idea because of the 6 ideas from Oraclus, this one was the most interesting. In addition, we wanted to learn how to make and train our NLP-related models, because we had never worked with NLP and ML before.

## Technology stack and organization

[List the technologies, frameworks and development processes used in your project.]

Technologies and Frameworks:

Flask
requests
yfinance
Plotly
numpy
pandas
scikit-learn
Keras
NLTK (Natural Language Toolkit)
Jinja2

## Solutions and features implemented

[Provide a detailed description of the solutions and features your team implemented. Include images if applicable. This section should be around 200-250 words.]

We have created a model (AI) to analyze the mood of the news. Our model is able to analyze Russian and English texts. The percentage of accuracy is ~80-83%. To train our model, we took 4 datasets from different areas: finance (5k rows), chatGPT generated news (1k rows), people's emotions on financial news from Twitter (500 rows), financial risks (6k rows). 
<img width="980" alt="Снимок экрана 2023-12-20 в 22 43 39" src="https://github.com/Futur1stXD/NewsAnalysis/assets/126179639/8fc2e5b6-e36c-43ca-9edb-823bda9426f6">
<img width="551" alt="Снимок экрана 2023-12-20 в 22 43 54" src="https://github.com/Futur1stXD/NewsAnalysis/assets/126179639/ffe2928e-b0e8-45b8-b74d-bf14f537013b">

We also wrote a function that compares the news mood with the market price after the news is released (it takes 10 minutes after the news is released). And she deduces whether this news has affected the market or not.
<img width="922" alt="Снимок экрана 2023-12-20 в 22 44 15" src="https://github.com/Futur1stXD/NewsAnalysis/assets/126179639/04b5b319-1bdf-4249-b9e9-59691d85d946">

For interactive purposes, we have created a website that shows our analysis of the news.
There is the main page where some crytocurrency tokens is shown with their real-time information.
![image](https://github.com/Futur1stXD/NewsAnalysis/assets/90721576/1dfef50a-e2bc-400a-8a0e-e10ad0de5979)
Then we can move to the details page where the graph of price over time is shown. Also you can see points on graph, this is news that has been published at that time. Under the graph there is information about news. It contains their links, published time and our prediction that we got from analysis.
![image](https://github.com/Futur1stXD/NewsAnalysis/assets/90721576/efd94d84-15bc-4184-a790-9d3e52ec074b)
![image](https://github.com/Futur1stXD/NewsAnalysis/assets/90721576/ecac1236-5b21-4acd-94b1-c63b1c5dc733)

## Challenges faced

[Discuss the challenges your team encountered during the development process.]

We were faced with the fact that there was no high-quality ready-made dataset, we had to make it ourselves. And we could not receive news for the last week or month, only today - this made the task very difficult.

## Lessons learned

[Share insights and lessons your team gained while working on the project.]

We learned how to train the NLP model and figured out how it works.

## Future work

[Outline potential future improvements or features your team would like to implement.]
We have already demonstrated to you that we can train our model in different languages, so we would like to add the main languages for news analysis (Chinese, Korean, etc.).
We could also train the model so that it can deduce the subjectivity of the news. And we could roughly train a model to predict exactly how much the price will fall due to the news. Also creating this project motivated us to learn more about ML and create more creative projects.

## Additional sources

[Include links to any additional sources of useful information that could benefit others interested in your project.]

