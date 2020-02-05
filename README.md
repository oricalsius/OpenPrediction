# OpenPrediction
Machine learning applied to quotations prediction

### UNDER ACTIVE DEVELOPMENT



## Description
The project aims to provide a machine learning API which is predicting the future quotation prices.

The API will work on a local server mode and provides REST capabilities to other applications like Excel.
There is another project *[CryptoExchangeAPI](../../../CryptoExchangeAPI/)* that will provide the Excel add-on to communicate with this API and enhance Excel capabilities.


## How it will work
1. The API will gather quotations from exchanges' APIs
2. Some indicators will be generated and added to quotations
3. The result will be saved in a local SQLite database
4. User can request these indicators or the price prediction (multiple machine learning methods will be provided)


## Contributions
Contribution to this project is open. Any kind of help is welcome. 

There is a lot to do. Here are some ideas that I am thinking about for future evolutions:
1. Create a web interface and a public website to bring this API to large public.
2. Give the user the ability to create and configure his own neural network in a simple manner using a web interface. Then save, the network description in a local file that can be inputed each time.

