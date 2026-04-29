from datetime import datetime
from pydantic import BaseModel, EmailStr, field_validator
from src.models.user import UserStatus


class RolResponse(BaseModel):
    id: int
    nombre: str

    model_config = {"from_attributes": True}


class UsuarioBase(BaseModel):
    email: EmailStr
    nombre_usuario: str
    nombre: str
    apellidos: str


class UsuarioCreate(UsuarioBase):
    password: str

    @field_validator("password")
    @classmethod
    def password_must_be_strong(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("La contraseña debe tener al menos 8 caracteres")
        return v

    @field_validator("nombre_usuario")
    @classmethod
    def nombre_usuario_must_be_valid(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError("El nombre de usuario debe tener al menos 3 caracteres")
        if not v.replace("_", "").replace(".", "").isalnum():
            raise ValueError("El nombre de usuario solo puede contener letras, números, puntos y guiones bajos")
        return v


class UsuarioUpdate(BaseModel):
    nombre_usuario: str | None = None
    nombre: str | None = None
    apellidos: str | None = None

    @field_validator("nombre_usuario")
    @classmethod
    def nombre_usuario_must_be_valid(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) < 3:
            raise ValueError("El nombre de usuario debe tener al menos 3 caracteres")
        if not v.replace("_", "").replace(".", "").isalnum():
            raise ValueError("El nombre de usuario solo puede contener letras, números, puntos y guiones bajos")
        return v


class PasswordUpdate(BaseModel):
    password_actual: str
    password_nuevo: str

    @field_validator("password_nuevo")
    @classmethod
    def password_must_be_strong(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("La contraseña debe tener al menos 8 caracteres")
        return v


class UsuarioResponse(UsuarioBase):
    id: int
    rol: RolResponse
    fecha_registro: datetime
    fecha_ultimo_acceso: datetime | None
    status: UserStatus

    model_config = {"from_attributes": True}


class UsuarioAdminResponse(UsuarioResponse):
    pass


class StatusUpdate(BaseModel):
    status: UserStatus