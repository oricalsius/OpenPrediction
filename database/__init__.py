from pathlib import Path

_DATABASE_PATH = str(Path(__file__).parent.absolute())
_DATABASE_STRING = 'sqlite:///' + _DATABASE_PATH + "/db.indicators.db"



