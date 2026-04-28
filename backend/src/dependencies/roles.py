from fastapi import Depends, HTTPException, status

from src.dependencies.auth import get_current_user
from src.models.user import User


async def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.rol.nombre != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso restringido a administradores",
        )
    return current_user