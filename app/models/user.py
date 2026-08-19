"""Schemas Pydantic del usuario autenticado."""

from typing import Any

from pydantic import BaseModel, Field


class CurrentUser(BaseModel):
    """Usuario autenticado, construido a partir de los claims de un ID token."""

    uid: str = Field(description="Identificador único del usuario en Firebase (claim `sub`).")
    email: str | None = Field(
        default=None,
        description=(
            "Email del usuario. Es `null` cuando el usuario entró por un método "
            "que no aporta email (teléfono o anónimo)."
        ),
    )
    email_verified: bool = Field(
        default=False,
        description="Si Firebase considera verificado el email.",
    )

    @classmethod
    def from_claims(cls, claims: dict[str, Any]) -> "CurrentUser":
        """Crea el usuario desde los claims ya verificados de un ID token."""
        return cls(
            uid=claims["sub"],
            email=claims.get("email"),
            email_verified=bool(claims.get("email_verified", False)),
        )