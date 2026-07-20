from fastapi import Request
from fastapi.responses import JSONResponse


class PredictionException(Exception):

    def __init__(self, message):

        self.message = message

async def prediction_exception_handler(request: Request,exc: PredictionException):

    return JSONResponse(

        status_code=500,

        content={

            "error": exc.message
        },
)