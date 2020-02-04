"""
Rest API Module defining the basic structure to communicate with Huobi API
The requirements have been gathered from Huobi API documentation: https://huobiapi.github.io/docs/spot/v1/en/
"""

from . import HuobiAPI
from exchangeapi.huobi.models.enumerations import TickerPeriod
from exchangeapi.huobi.utils import (get_json_key, check_str, check_range)
from urllib.parse import (urlparse, urlunparse, urlencode, parse_qs)
import json
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

    def _format_uri(self, url: str, parameters: dict = None) -> str:
        # Parse url and add the netloc part
        parsed_url = list(urlparse(super().global_uri))
        parsed_url[2] = url

        if parameters is not None:
            # Parse uri parameters and sort them
            query = parse_qs(parsed_url[4])
            query.update(parameters)
            query = dict(sorted(query.items(), key=lambda x: x[0]))
            parsed_url[4] = urlencode(query)

        return urlunparse(parsed_url)

    async def _http_get_request(self, uri: str, parameters: dict) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(uri, params=parameters, timeout=self.timeout) as resp:
                json_body = await resp.json(content_type=None)
                if resp.status != 200:
                    raise Exception(f"Error while requesting data from the server. Response status: {resp.status}")
                return json_body

    async def get_ticker_history(self, symbol: str, period: TickerPeriod, size: int = 1, schema: object = None,
                                 **kwargs_schema) -> int:
        """
        Retrieves all klines in a specific range.

        Implements end point https://api.huobi.pro/market/history/kline

        :param str symbol: The trading symbol to query.
        :param TickerPeriod period: The period of each candle.
        :param int size: The number of data returns in range [0, 2000]
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
        json_object = get_json_key(root_json, "data")

        if schema is None:
            return json_object
        else:
            res = schema.load(json_object, **kwargs_schema)
            return res

