from __future__ import annotations

import base64
import hashlib
import os
from datetime import UTC, datetime

from cryptography.fernet import Fernet

from appliance.db_ext import WorkbenchDB


def derive_key(master_secret: str) -> bytes:
    digest = hashlib.sha256(master_secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)


class SecretStore:
    def __init__(self, db: WorkbenchDB, master_secret: str) -> None:
        self.db = db
        self.fernet = Fernet(derive_key(master_secret))

    def set(self, name: str, value: str) -> None:
        ciphertext = self.fernet.encrypt(value.encode())
        now = datetime.now(UTC).isoformat()
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO secrets(name,ciphertext,created_at,updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(name) DO UPDATE SET
                  ciphertext=excluded.ciphertext,
                  updated_at=excluded.updated_at
                """,
                (name, ciphertext, now, now),
            )

    def get(self, name: str) -> str | None:
        with self.db.connect() as con:
            row = con.execute(
                "SELECT ciphertext FROM secrets WHERE name=?", (name,)
            ).fetchone()
        if row is None:
            return None
        return self.fernet.decrypt(row["ciphertext"]).decode()
