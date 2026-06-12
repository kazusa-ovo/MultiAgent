from __future__ import annotations

import json
from pathlib import Path
from typing import Any

class UserProfile:
    def __init__(self,storage_path: str = "data/user_profile.json"):
        self._path = Path(storage_path)
        self._data: dict[str, Any] = {}
        self._load()
    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path,"r",encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self,key: str,default: Any = None) -> Any:
        return self._data.get(key,default)

    def set(self,key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def delete(self,key: str) -> None:
        self._data.pop(key,None)
        self.save()

    @property
    def preferences(self) -> dict[str,Any]:
        return self._data.setdefault("preferences",{})

    def set_preferences(self,key: str, value: Any) -> None:
        prefs = self._data.setdefault("preferences",{})
        prefs[key] = value
        self.save()

    def to_context_string(self) -> str:
        parts: list[str] = []
        if self._data.get("preferences"):
            prefs = ", ".join(
                f"{k}={v}" for k,v in self._data["preferences"].items()
            )
            parts.append(f"User preferences: {prefs}")
        if self._data.get("known_facts"):
            facts = ", ".join(
                f"{k}: {v}" for k, v in self._data["known_facts"].items()
            )
            parts.append(f"Known facts: {facts}")
        return "\n".join(parts)
