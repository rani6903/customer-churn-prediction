import time

from starlette.middleware.base import BaseHTTPMiddleware

from src.logger import logger


class LoggingMiddleware(
    BaseHTTPMiddleware
):

    async def dispatch(
        self,
        request,
        call_next,
    ):

        start = time.time()

        response = await call_next(
            request
        )

        process_time = (
            time.time()
            - start
        )

        logger.info(

            f"{request.method} "

            f"{request.url.path} "

            f"{response.status_code} "

            f"{process_time:.3f}s"

        )

        return response