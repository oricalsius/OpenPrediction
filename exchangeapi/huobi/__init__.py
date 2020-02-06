from abc import ABC, abstractmethod


class HuobiAPI(ABC):
    """Represent the basic abstract class from which Rest and WebSocket Huobi API should inherit"""

    _URL = "{0}://api.huobi.pro{1}"
    _URL_AWS = "{0}://api-aws.huobi.pro{1}"
    _SIGNATURE_METHOD = "HmacSHA256"
    _SIGNATURE_VERSION = "2"
    _URL_COMMON = "/v1/common/"
    _URL_MARKET_DATA = "/market/"
    _URL_REFERENCE = "/v2/reference/"

    @abstractmethod
    def __init__(self, access_key: str = "", secret_key: str = "", use_aws: bool = False, is_socket: bool = False):
        self._is_websocket = is_socket
        self._use_aws = use_aws
        self.access_key = access_key
        self.secret_key = secret_key

        self._set_global_uri()

    def _set_global_uri(self):
        if self.use_aws:
            self._global_uri = HuobiAPI._URL_AWS
        else:
            self._global_uri = HuobiAPI._URL

        if self.is_websocket:
            self._global_uri = self._global_uri.format("wss", "/ws")
        else:
            self._global_uri = self._global_uri.format("https", "")

    @property
    def use_aws(self) -> bool:
        return self._use_aws

    @use_aws.setter
    def use_aws(self, value: bool):
        self._use_aws = value
        self._set_global_uri()

    @property
    def is_websocket(self) -> bool:
        return self._is_websocket

    @is_websocket.setter
    def is_websocket(self, value: bool):
        self._is_websocket = value
        self._set_global_uri()

    @property
    def global_uri(self) -> str:
        return self._global_uri
