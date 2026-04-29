from datetime import UTC, datetime, timedelta

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
    verify_token_hash,
    decode_token,
)
from src.models.token import LogActividad, RefreshToken
from src.repositories.user import UserRepository
from src.schemas.token import TokenResponse
from fastapi import HTTPException, status
from jose import JWTError

settings = get_settings()


class AuthService:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_repo = UserRepository(db)

    async def register(self, email: str, password: str, nombre_usuario: str,
                       nombre: str, apellidos: str, request: Request) -> TokenResponse:
        if await self.user_repo.get_by_email(email):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="El email ya está registrado",
            )

        if await self.user_repo.get_by_nombre_usuario(nombre_usuario):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="El nombre de usuario ya está en uso",
            )

        rol = await self.user_repo.get_rol_by_nombre("user")
        if not rol:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno: rol por defecto no encontrado",
            )

        usuario = await self.user_repo.create(
            email=email,
            password_hash=hash_password(password),
            nombre_usuario=nombre_usuario,
            nombre=nombre,
            apellidos=apellidos,
            rol_id=rol.id,
        )

        tokens = await self._create_tokens(usuario.id, request)

        await self._log(usuario.id, "registro", request)
        await self.db.commit()

        return tokens

    async def login(self, email: str, password: str,
                    request: Request) -> TokenResponse:
        error_generico = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
        )

        usuario = await self.user_repo.get_by_email(email)
        if not usuario or not verify_password(password, usuario.password_hash):
            raise error_generico

        if usuario.status.value != "activo":
            raise error_generico

        await self.user_repo.update(usuario, fecha_ultimo_acceso=datetime.now(UTC))

        tokens = await self._create_tokens(usuario.id, request)

        await self._log(usuario.id, "login", request)
        await self.db.commit()

        return tokens

    async def logout(self, refresh_token: str, usuario_id: int,
                     request: Request) -> None:
        token_obj = await self._find_active_token(refresh_token, usuario_id)
        if token_obj:
            token_obj.activo = False
            await self.db.flush()

        await self._log(usuario_id, "logout", request)
        await self.db.commit()

    async def refresh(self, refresh_token: str) -> TokenResponse:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
        )

        try:
            payload = decode_token(refresh_token)
            if payload.get("type") != "refresh":
                raise credentials_exception
            usuario_id = int(payload.get("sub"))
        except (JWTError, TypeError, ValueError):
            raise credentials_exception

        token_obj = await self._find_active_token(refresh_token, usuario_id)
        if not token_obj:
            raise credentials_exception

        usuario = await self.user_repo.get_by_id(usuario_id)
        if not usuario or usuario.status.value != "activo":
            raise credentials_exception

        token_obj.activo = False
        await self.db.flush()

        from fastapi import Request
        class _FakeRequest:
            def __init__(self):
                self.client = None
                self.headers = {}

        fake_req = _FakeRequest()
        new_tokens = await self._create_tokens(usuario_id, fake_req)
        await self.db.commit()

        return new_tokens

    async def revoke_all_tokens(self, usuario_id: int, request: Request) -> None:
        from sqlalchemy import update
        from src.models.token import RefreshToken

        await self.db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.usuario_id == usuario_id,
                RefreshToken.activo == True,
            )
            .values(activo=False)
        )

        await self._log(usuario_id, "logout", request)
        await self.db.commit()

    async def _create_tokens(self, usuario_id: int, request) -> TokenResponse:
        access_token = create_access_token(str(usuario_id))
        refresh_token = create_refresh_token(str(usuario_id))

        ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent") if hasattr(request, "headers") else None

        expire = datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        token_obj = RefreshToken(
            usuario_id=usuario_id,
            token_hash=hash_token(refresh_token),
            fecha_expiracion=expire,
            activo=True,
            ip_origen=ip,
            user_agent=user_agent,
        )
        self.db.add(token_obj)
        await self.db.flush()

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )

    async def _find_active_token(self, refresh_token: str,
                                  usuario_id: int) -> RefreshToken | None:
        from sqlalchemy import select
        from src.models.token import RefreshToken as RT

        result = await self.db.execute(
            select(RT).where(
                RT.usuario_id == usuario_id,
                RT.activo == True,
                RT.fecha_expiracion > datetime.now(UTC),
            )
        )
        tokens = result.scalars().all()

        for token_obj in tokens:
            if verify_token_hash(refresh_token, token_obj.token_hash):
                return token_obj

        return None

    async def _log(self, usuario_id: int, accion: str, request) -> None:
        ip = request.client.host if request.client else None
        log = LogActividad(
            usuario_id=usuario_id,
            accion=accion,
            ip=ip,
        )
        self.db.add(log)
        await self.db.flush()