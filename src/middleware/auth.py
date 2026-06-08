import logging
from os import getenv

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

SECRET_KEY = getenv("SECRET_KEY")
ALGORITHM = getenv("ALGORITHM", "HS256")


async def auth_middleware(request: Request, call_next):
    public_paths = {
        "/running",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    if request.url.path in public_paths:
        return await call_next(request)

    auth_header = request.headers.get("Authorization")

    logger.info(f"Authorization: {auth_header}")
    logger.info(f"SECRET_KEY configured: {SECRET_KEY is not None}")
    logger.info(f"ALGORITHM: {ALGORITHM}")

    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )

    try:
        scheme, token = auth_header.split(" ", 1)

        logger.info(f"Scheme: {scheme}")

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        request.state.user = payload

    except Exception as e:
        logger.exception(e)
        raise

    return await call_next(request)
