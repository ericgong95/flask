class InvalidUsage(Exception):
    status_code = 400

    def __init__(self,msg,status_code=None,payload=None):
        Exception.__init__(self)
        self.message = msg
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
    