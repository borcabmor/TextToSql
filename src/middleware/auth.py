from os import getenv

from fastapi import Request, status
from fastapi.responses import JSONResponse
from jose import JWTError, jwt

SECRET_KEY = getenv("SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set")

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

    if not auth_header:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Authorization header missing"},
        )

    try:
        scheme, token = auth_header.split(" ", 1)

        if scheme.lower() != "bearer":
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid auth scheme"},
            )

        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": True}
        )

        if payload.get("project_code") != "T2SQL":
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "System incorrect."},
            )

        request.state.user = payload

    except JWTError as e:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": f"JWT error: {str(e)}"},
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": f"Auth error: {str(e)}"},
        )

    return await call_next(request)
