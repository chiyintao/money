import json, logging, sys, time

class JsonFormatter(logging.Formatter):
    def format(self,record): return json.dumps({'timestamp':time.time(),'level':record.levelname,'event':getattr(record,'event_name',record.getMessage()),'message':record.getMessage()},ensure_ascii=False)

def logger(name='paper'): return logging.getLogger(name)

def configure():
    handler=logging.StreamHandler(sys.stdout); handler.setFormatter(JsonFormatter()); root=logging.getLogger('paper'); root.setLevel(logging.INFO); root.handlers.clear(); root.addHandler(handler); return root
