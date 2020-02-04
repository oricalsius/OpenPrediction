import os
from aiohttp import web
import json

DATABASE_PATH = "./database/db.indicators.db"

app = web.Application()

async def handle(request):
    response_obj = { 'status' : 'success' }
    return web.Response(text=json.dumps(response_obj), status=200)


#app.route.add_get('/', handle)
#app.route.add_post('/', handle)

#app.on_startup.append(handle)
#app.on_cleanup.append(handle)
#app.on_shutdown.append(handle)
#web.run_app(app)





# Init flask
flask_app = Flask(__name__)

# aaaa config
flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DATABASE_PATH
flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#Init aaaa
db = SQLAlchemy(flask_app)




#Indicators.query.all()
#Quotations.query.all()
#Quotations.query.get(variable)
#db.session.add(new Indicator)
#db.session.commit()
#db.session.delete(quotation)



