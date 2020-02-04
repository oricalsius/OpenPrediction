from .model import (Quotation, Indicators, get_base)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from . import _DATABASE_STRING


class DbConnexion:
    """
    Basic class to handle connexion to the database and to provide the necessary functions to modify data in our
    database
    """

    def __init__(self, db_path:str = None, drop_all: bool = True, create_all: bool = True, echo: bool = False):
        if db_path is not None and db_path.strip() != "":
            self._DATABASE_STRING = db_path
        else:
            self._DATABASE_STRING = _DATABASE_STRING

        self._Base = get_base()
        self._engine = create_engine(self._DATABASE_STRING, echo=echo)   # Declare engine
        self._session = sessionmaker(bind=self._engine)

        # Drop old database
        if drop_all:
            self._Base.metadata.drop_all(self._engine)

        # Create new database or tables only if it already exists
        # Previous data are not touched
        if create_all:
            self._Base.metadata.create_all(self._engine)

    @property
    def get_session(self):
        return self._session

    @get_session.setter
    def get_session(self, session_obj: object):
        raise AttributeError("DbConnexion.session attribute is read only. You do not have rights to alter it.")

