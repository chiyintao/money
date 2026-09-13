class ConnectorHealth:
    def __init__(self, name):
        self.name = name
        self.connected = False
        self.reconnects = 0
        self.events = 0
        self.last_event_time = None
        self.last_error = None
        self.stream_times = {}

    def observe(self, event_time, stream='default'):
        event_time = int(event_time)
        previous = self.stream_times.get(stream)
        if previous is not None and event_time < previous:
            self.last_error = 'event_time_regression'
            return False
        self.connected = True
        self.events += 1
        self.stream_times[stream] = event_time
        self.last_event_time = max(event_time, self.last_event_time or 0)
        self.last_error = None
        return True

    def disconnected(self, error=None):
        self.connected = False
        self.reconnects += 1
        self.last_error = repr(error) if error else self.last_error

    def snapshot(self):
        return {'name': self.name, 'connected': self.connected, 'reconnects': self.reconnects, 'events': self.events, 'last_event_time': self.last_event_time, 'last_error': self.last_error}
