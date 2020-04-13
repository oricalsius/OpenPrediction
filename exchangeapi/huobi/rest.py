"""
Rest API Module defining the basic structure to communicate with Huobi API
The requirements have been gathered from Huobi API documentation: https://huobiapi.github.io/docs/spot/v1/en/
"""

from . import HuobiAPI
from exchangeapi.huobi.models.enumerations import TickerPeriod
from exchangeapi.huobi.utils import (get_json_key, check_str, check_range, parse_json_to_object)
from urllib.parse import (urlparse, urlunparse, urlencode, parse_qs)
from typing import Any

import json
import asyncio
import datetime
import aiohttp


class Rest(HuobiAPI):

    """
    Rest API class handling communication with Huobi server to get access to all needed information.

    Attributes:
        access_key (str): User access key generated in API management.
        secret_key (str): User secret key generated in API management and used to hash url.
        use_aws (bool): Whether to use aws huobi server or not.
        timeout (int): Timeout before raising an error.
    """

    def __init__(self, access_key: str = "", secret_key: str = "", use_aws: bool = False,
                 timeout: int = -1):
        super().__init__(access_key, secret_key,  use_aws, is_socket=False)
        #self.session = aiohttp.ClientSession()
        self.timeout = aiohttp.ClientTimeout()
        if timeout > 0:
            self.timeout = aiohttp.ClientTimeout(total=timeout)

    def __del__(self):
        #self.session.close()
        pass

    def _format_uri(self, url: str, parameters: dict = None, global_uri: str = None) -> str:
        # Parse url and add the netloc part
        global_uri = super().global_uri if (global_uri is None) or (global_uri == '') else global_uri
        parsed_url = list(urlparse(global_uri))
        parsed_url[2] = url

        if parameters is not None:
            # Parse uri parameters and sort them
            query = parse_qs(parsed_url[4])
            query.update(parameters)
            query = dict(sorted(query.items(), key=lambda x: x[0]))
            parsed_url[4] = urlencode(query)

        return urlunparse(parsed_url)

    async def _http_get_request(self, uri: str, parameters: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(uri, params=parameters, timeout=self.timeout) as resp:
                json_body = await resp.json(content_type=None)
                if resp.status != 200:
                    raise Exception(f"Error while requesting data from the server. Response status: {resp.status}")
                return json_body

    async def get_ticker_history_async(self, symbol: str, period: TickerPeriod, size: int = 1, schema: object = None,
                                       kwargs_schema: dict = dict()) -> Any:
        """
        Retrieves all klines in a specific range.

        Async version.

        Implements end point https://api.huobi.pro/market/history/kline

        :param str symbol: The trading symbol to query.
        :param TickerPeriod period: The period of each candle.
        :param int size: The number of data returned in range [0, 2000]
        :param object schema: Schema to apply to parse json. (str, dict, MarshmallowSchema)
        :param dict kwargs_schema: Key word arguments to pass to schema.
        :return: Last n tickers for a specific period parsed to schema
        :rtype: Any(str, dict(Any, Any), class)
        """
        check_range("size", size, [1, 2001])
        check_str("symbol", symbol)

        parameters = {"symbol": symbol, "period": period, "size": size}
        uri = self._format_uri(super()._URL_MARKET_DATA + "history/kline")

        root_json = await self._http_get_request(uri, parameters)
        return parse_json_to_object(get_json_key(root_json, "data"), schema, **kwargs_schema)

    def get_ticker_history(self, symbol: str, period: TickerPeriod, size: int = 1, schema: object = None,
                           kwargs_schema: dict = dict()) -> Any:

        return asyncio.run(self.get_ticker_history_async(symbol, period, size, schema, kwargs_schema))

    # Temporary, will be improved to respect a best coding principle
    async def get_ticker_history_cryptocompare_async(self, fsym: str, tsym: str, api_key: str, to_date: str = None,
                                                     schema: object = None, kwargs_schema: dict = dict()) -> Any:

        check_str("symbol", fsym)
        check_str("tsym", fsym)
        check_str("api_key", fsym)

        uri = self._format_uri("data/v2/histohour", global_uri="https://min-api.cryptocompare.com/")
        res = []

        # Compute initial toTs
        if to_date is None:
            to_date = datetime.datetime.now()
        else:
            to_date = datetime.datetime.strptime(to_date, "%Y-%m-%d")

        from_date = datetime.datetime.now()
        toTs = int(from_date.timestamp())
        diff = from_date - to_date
        days, seconds = diff.days, diff.seconds
        diff = days * 24 + seconds // 3600
        while diff >= 1:
            limit = min(int(diff), 2000)

            parameters = {"fsym": fsym, "tsym": tsym, "e": "HuobiPro", "toTs": toTs, "limit": limit,
                          "api_key": api_key}

            root_json = await self._http_get_request(uri, parameters)
            data = get_json_key(root_json, "$.Data.Data")
            res.extend(data)

            from_date = datetime.datetime.fromtimestamp(data[0]["time"])
            toTs = int(from_date.timestamp())
            diff = from_date - to_date
            days, seconds = diff.days, diff.seconds
            diff = days * 24 + seconds // 3600

        return parse_json_to_object(res, schema, **kwargs_schema)

    def get_ticker_history_cryptocompare(self, fsym: str, tsym: str, api_key: str, to_date: str = None,
                                         schema: object = None, kwargs_schema: dict = dict()) -> Any:

        return asyncio.run(self.get_ticker_history_cryptocompare_async(fsym, tsym, api_key, to_date, schema,
                                                                       kwargs_schema))


