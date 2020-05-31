
CONFIG = {
    "COSMOS_DB":
        {
            "CONNECTION_STRING": "mongodb://{username}:{password}@{host}:{port}/?ssl={ssl}&replicaSet=globaldb" +
                                 "&retrywrites=false&maxIdleTimeMS=120000&appName=@{appname}@",
            "USERNAME": "username",
            "PASSWORD": "password",
            "HOST": "host",
            "PORT": "port",
            "SSL": "true",
            "APPNAME": "app_name",
        },

    "CRYPTOCOMPARE_KEY": "api_key"
}

