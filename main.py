from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
import time
from math import ceil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

try:
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover - fallback for older AstrBot builds
    TextPart = None

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
except Exception:  # pragma: no cover - fallback for older AstrBot builds
    get_astrbot_plugin_data_path = None


PLUGIN_NAME = "astrbot_plugin_multi_bot_control"
LOCAL_BOTS_FILE = "group_bots.json"
PRIORITY_FILE = "controlled_priorities.json"


@dataclass
class BotEntry:
    qq: str = ""
    name: str = ""
    call_name: str = ""
    kind: str = "uncontrolled"
    source: str = "global"
    allow_interaction: bool = True

    @property
    def identity(self) -> str:
        return f"qq:{self.qq}"

    @property
    def display_name(self) -> str:
        return self.name or self.call_name or self.qq or "unknown-bot"

    @property
    def effective_call_name(self) -> str:
        return self.call_name or self.name or self.qq or "对方机器人"

    def to_json(self) -> dict[str, Any]:
        return {
            "qq": self.qq,
            "name": self.name,
            "call_name": self.call_name,
            "kind": self.kind,
            "source": self.source,
            "allow_interaction": self.allow_interaction,
        }


@dataclass
class BotSessionState:
    turns: int = 0
    cooldown_until: float = 0.0
    last_accepted_at: float = 0.0
    recent_messages: list[tuple[float, str]] = field(default_factory=list)


def _unique_non_empty(values: list[Any]) -> list[str]:
    ret: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            ret.append(text)
    return ret


def _normalize_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _stable_message_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _stable_priority(identity: str) -> int:
    digest = hashlib.sha256(f"{PLUGIN_NAME}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 1_000_000 + 1


def _component_type_name(component: Any) -> str:
    typ = getattr(component, "type", "")
    return str(typ).lower()


@register(
    PLUGIN_NAME,
    "OpenCode",
    "控制群聊中可识别机器人之间的有限交互，避免循环调用。",
    "0.2.1",
)
class MultiBotControlPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.data_dir = self._resolve_data_dir()
        self.group_bots_path = self.data_dir / LOCAL_BOTS_FILE
        self.priority_path = self.data_dir / PRIORITY_FILE
        self.sessions: dict[str, BotSessionState] = {}
        self.group_window_times: dict[str, list[float]] = {}
        self.no_human_times: dict[str, list[float]] = {}
        self.no_human_cooldowns: dict[str, float] = {}
        self._group_bots_cache: dict[str, Any] | None = None
        self._priority_cache: dict[str, int] | None = None

    async def initialize(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_controlled_priorities()

    def _resolve_data_dir(self) -> Path:
        if get_astrbot_plugin_data_path:
            return Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        return Path("data") / "plugin_data" / PLUGIN_NAME

    def _enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _section(self, name: str) -> dict[str, Any]:
        value = self.config.get(name, {})
        return value if isinstance(value, dict) else {}

    def _limit_int(self, key: str, default: int, minimum: int = 0) -> int:
        value = self._section("limits").get(key, default)
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return default

    def _conf_bool(self, section: str, key: str, default: bool) -> bool:
        return bool(self._section(section).get(key, default))

    def _conf_int(self, section: str, key: str, default: int, minimum: int = 0) -> int:
        value = self._section(section).get(key, default)
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return default

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"{PLUGIN_NAME}: 读取 {path} 失败: {e}")
            return default

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def _load_group_bots_data(self) -> dict[str, Any]:
        if self._group_bots_cache is None:
            data = self._read_json(self.group_bots_path, {"groups": {}})
            if not isinstance(data, dict):
                data = {"groups": {}}
            data.setdefault("groups", {})
            self._group_bots_cache = data
        return self._group_bots_cache

    def _save_group_bots_data(self) -> None:
        if self._group_bots_cache is not None:
            self._write_json(self.group_bots_path, self._group_bots_cache)

    def _load_priorities(self) -> dict[str, int]:
        if self._priority_cache is None:
            data = self._read_json(self.priority_path, {})
            if not isinstance(data, dict):
                data = {}
            self._priority_cache = {
                str(k): int(v)
                for k, v in data.items()
                if isinstance(v, int) or str(v).isdigit()
            }
        return self._priority_cache

    def _save_priorities(self) -> None:
        if self._priority_cache is not None:
            self._write_json(self.priority_path, self._priority_cache)

    def _normalize_bot_entry(self, raw: Any, kind: str, source: str) -> BotEntry | None:
        if isinstance(raw, str):
            parts = [part.strip() for part in re.split(r"[,，]", raw) if part.strip()]
            raw = {
                "qq": parts[0] if parts else "",
                "name": parts[1] if len(parts) > 1 else "",
                "call_name": parts[2] if len(parts) > 2 else "",
            }
        if not isinstance(raw, dict):
            return None
        qq = _normalize_id(raw.get("qq") or raw.get("id") or raw.get("user_id"))
        name = str(raw.get("name") or raw.get("nickname") or "").strip()
        call_name = str(raw.get("call_name") or raw.get("callname") or "").strip()
        allow_interaction = bool(raw.get("allow_interaction", True))
        kind = str(raw.get("kind") or kind or "uncontrolled").strip() or "uncontrolled"
        if kind not in {"controlled", "uncontrolled"}:
            kind = "uncontrolled"
        if not qq:
            return None
        return BotEntry(
            qq=qq,
            name=name,
            call_name=call_name,
            kind=kind,
            source=source,
            allow_interaction=allow_interaction,
        )

    def _global_entries(self) -> list[BotEntry]:
        registry = self._section("bot_registry")
        entries: list[BotEntry] = []
        for raw in registry.get("controlled_bots", []) or []:
            entry = self._normalize_bot_entry(raw, "controlled", "global")
            if entry:
                entries.append(entry)
        for raw in registry.get("uncontrolled_bots", []) or []:
            entry = self._normalize_bot_entry(raw, "uncontrolled", "global")
            if entry:
                entries.append(entry)
        return entries

    def _local_entries(self, group_id: str) -> list[BotEntry]:
        data = self._load_group_bots_data()
        group_data = data.get("groups", {}).get(str(group_id), {})
        entries = []
        for raw in group_data.get("bots", []) or []:
            entry = self._normalize_bot_entry(raw, "uncontrolled", "group")
            if entry:
                entry.kind = "uncontrolled"
                entry.source = "group"
                entries.append(entry)
        return entries

    def _effective_entries(self, group_id: str) -> list[BotEntry]:
        merged: dict[str, BotEntry] = {}
        for entry in self._global_entries():
            merged[entry.identity] = entry
        for entry in self._local_entries(group_id):
            merged[entry.identity] = entry
        return list(merged.values())

    def _ensure_controlled_priorities(self) -> None:
        priorities = self._load_priorities()
        changed = False
        for entry in self._global_entries():
            if entry.kind != "controlled":
                continue
            key = entry.identity
            if key in priorities:
                continue
            priority = _stable_priority(key)
            used = set(priorities.values())
            while priority in used:
                priority = priority % 1_000_000 + 1
            priorities[key] = priority
            changed = True
        if changed:
            self._save_priorities()

    def _priority(self, entry: BotEntry) -> int:
        priorities = self._load_priorities()
        if entry.identity not in priorities:
            self._ensure_controlled_priorities()
        return int(priorities.get(entry.identity, 500_000))

    def _bot_log_id(self, entry: BotEntry) -> str:
        return entry.qq or entry.display_name or entry.identity

    def _remaining_session_turns(self, state: BotSessionState) -> str:
        limits = self._section("limits")
        if not bool(limits.get("enable_max_turns_per_session", True)):
            return "不限"
        max_turns = self._limit_int("max_turns_per_session", 2, minimum=1)
        return str(max(0, max_turns - state.turns))

    def _remaining_block_seconds(self, state: BotSessionState, now: float) -> int:
        if state.cooldown_until > now:
            return max(0, ceil(state.cooldown_until - now))
        return 0

    def _remaining_group_block_seconds(self, group_id: str, now: float) -> int:
        cooldown_until = self.no_human_cooldowns.get(group_id, 0.0)
        if cooldown_until > now:
            return max(0, ceil(cooldown_until - now))
        return 0

    def _log_controlled_reply_blocked(
        self,
        self_entry: BotEntry,
        state: BotSessionState,
        reason: str,
        now: float,
    ) -> None:
        if self_entry.kind != "controlled" or self_entry.source != "global":
            return
        logger.info(
            f"机器人 {self._bot_log_id(self_entry)} 的对 bot 回复冷却中，原因：{reason}，剩余时间：{self._remaining_block_seconds(state, now)} 秒",
        )

    def _log_group_reply_blocked(
        self,
        event: AstrMessageEvent,
        self_entry: BotEntry,
        state: BotSessionState,
        reason: str,
        now: float,
    ) -> None:
        if self_entry.kind != "controlled" or self_entry.source != "global":
            return
        group_id = _normalize_id(event.get_group_id()) or "nogroup"
        remaining = self._remaining_group_block_seconds(group_id, now)
        if remaining <= 0:
            remaining = self._remaining_block_seconds(state, now)
        logger.info(
            f"机器人 {self._bot_log_id(self_entry)} 的对 bot 回复冷却中，原因：{reason}，剩余时间：{remaining} 秒",
        )

    def _find_peer_bot(self, event: AstrMessageEvent, entries: list[BotEntry]) -> BotEntry | None:
        sender_id = _normalize_id(event.get_sender_id())
        if sender_id:
            for entry in entries:
                if entry.qq and entry.qq == sender_id:
                    return entry
        return None

    def _self_entry(self, event: AstrMessageEvent, entries: list[BotEntry]) -> BotEntry:
        self_id = _normalize_id(event.get_self_id())
        for entry in entries:
            if entry.kind == "controlled" and entry.qq and entry.qq == self_id:
                return entry
        return BotEntry(qq=self_id, name=self_id, call_name=self_id, kind="controlled", source="runtime")

    def _at_targets(self, event: AstrMessageEvent) -> set[str]:
        targets: set[str] = set()
        for component in event.get_messages():
            if isinstance(component, Comp.At) or _component_type_name(component).endswith("at"):
                qq = _normalize_id(getattr(component, "qq", ""))
                if qq and qq != "all":
                    targets.add(qq)
        return targets

    def _has_at_all(self, event: AstrMessageEvent) -> bool:
        for component in event.get_messages():
            if isinstance(component, getattr(Comp, "AtAll", Comp.At)):
                return True
            if isinstance(component, Comp.At) and _normalize_id(getattr(component, "qq", "")) == "all":
                return True
        return False

    def _reply_sender_ids(self, event: AstrMessageEvent) -> set[str]:
        sender_ids: set[str] = set()
        for component in event.get_messages():
            if isinstance(component, Comp.Reply) or _component_type_name(component).endswith("reply"):
                sender_id = _normalize_id(getattr(component, "sender_id", ""))
                if sender_id:
                    sender_ids.add(sender_id)
        return sender_ids

    def _message_targets_entry(self, event: AstrMessageEvent, entry: BotEntry) -> bool:
        target = self._section("targeting")
        at_targets = self._at_targets(event)
        if entry.qq and entry.qq in at_targets:
            return True
        if bool(target.get("enable_reply_target", True)) and entry.qq in self._reply_sender_ids(event):
            return True
        if bool(target.get("treat_at_all_as_target", False)) and self._has_at_all(event):
            return True
        return False

    def _wake_prefix_only_targets_self(self, event: AstrMessageEvent, self_entry: BotEntry) -> bool:
        target = self._section("targeting")
        if not bool(target.get("treat_wake_prefix_as_target", False)):
            return False
        if self._message_targets_entry(event, self_entry):
            return False
        return bool(event.is_at_or_wake_command)

    def _targeted_controlled_entries(
        self,
        event: AstrMessageEvent,
        entries: list[BotEntry],
    ) -> list[BotEntry]:
        controlled = [entry for entry in entries if entry.kind == "controlled"]
        return [entry for entry in controlled if self._message_targets_entry(event, entry)]

    def _self_rank_delay(
        self,
        self_entry: BotEntry,
        targeted_controlled: list[BotEntry],
    ) -> float:
        multi = self._section("multi_controlled")
        if not bool(multi.get("enable_priority_order", True)):
            return 0.0
        if len(targeted_controlled) <= 1:
            return 0.0
        reverse = multi.get("priority_order", "small_first") == "large_first"
        sorted_entries = sorted(targeted_controlled, key=self._priority, reverse=reverse)
        identities = [entry.identity for entry in sorted_entries]
        if self_entry.identity not in identities:
            return 0.0
        rank = identities.index(self_entry.identity)
        spacing = self._conf_int("multi_controlled", "controlled_reply_spacing_seconds", 2)
        return float(rank * spacing)

    def _session_key(self, event: AstrMessageEvent, peer: BotEntry) -> str:
        return ":".join(
            [
                _normalize_id(event.get_group_id()) or "nogroup",
                _normalize_id(event.get_self_id()) or "noself",
                peer.identity,
            ],
        )

    def _state_for(self, key: str) -> BotSessionState:
        if key not in self.sessions:
            self.sessions[key] = BotSessionState()
        return self.sessions[key]

    def _reset_group_turns(self, group_id: str) -> None:
        group_prefix = f"{group_id}:"
        for key, state in self.sessions.items():
            if key.startswith(group_prefix):
                state.turns = 0
                state.recent_messages.clear()
        self.no_human_times[group_id] = []
        self.no_human_cooldowns[group_id] = 0.0

    def _prune_times(self, values: list[float], now: float, window: int) -> list[float]:
        if window <= 0:
            return []
        return [ts for ts in values if now - ts <= window]

    def _deny_with_cooldown(self, state: BotSessionState, now: float) -> None:
        limits = self._section("limits")
        if bool(limits.get("enable_cooldown_after_limit", True)):
            state.cooldown_until = now + self._limit_int("cooldown_seconds", 300)

    def _deny_without_human_with_cooldown(self, group_id: str, now: float) -> None:
        self.no_human_cooldowns[group_id] = now + self._limit_int("without_human_cooldown_seconds", 300)

    def _can_accept_bot_request(
        self,
        event: AstrMessageEvent,
        peer: BotEntry,
        state: BotSessionState,
        now: float,
    ) -> tuple[bool, str]:
        if not peer.allow_interaction:
            return False, "peer interaction disabled"

        if state.cooldown_until and state.cooldown_until <= now:
            state.cooldown_until = 0.0
            state.turns = 0
        if state.cooldown_until > now:
            return False, "peer in cooldown"

        limits = self._section("limits")
        if bool(limits.get("enable_min_interval_per_peer", True)):
            min_interval = self._limit_int("min_interval_per_peer_seconds", 10)
            if state.last_accepted_at and now - state.last_accepted_at < min_interval:
                return False, "peer interval limit"

        if bool(limits.get("enable_max_turns_per_session", True)):
            max_turns = self._limit_int("max_turns_per_session", 2, minimum=1)
            if state.turns >= max_turns:
                self._deny_with_cooldown(state, now)
                return False, "session turn limit"

        group_id = _normalize_id(event.get_group_id()) or "nogroup"
        if bool(limits.get("enable_without_human_limit", True)):
            cooldown_until = self.no_human_cooldowns.get(group_id, 0.0)
            if cooldown_until and cooldown_until <= now:
                self.no_human_cooldowns[group_id] = 0.0
                self.no_human_times[group_id] = []
            elif cooldown_until > now:
                return False, "without human cooldown"

        if bool(limits.get("enable_group_window_limit", True)):
            window = self._limit_int("group_window_seconds", 300, minimum=1)
            max_group = self._limit_int("max_bot_replies_per_group_window", 6, minimum=1)
            times = self._prune_times(self.group_window_times.get(group_id, []), now, window)
            self.group_window_times[group_id] = times
            if len(times) >= max_group:
                return False, "group window limit"

        if bool(limits.get("enable_without_human_limit", True)):
            window = self._limit_int("without_human_window_seconds", 300, minimum=1)
            max_without_human = self._limit_int("max_bot_replies_without_human", 4, minimum=1)
            times = self._prune_times(self.no_human_times.get(group_id, []), now, window)
            self.no_human_times[group_id] = times
            if len(times) >= max_without_human:
                self._deny_without_human_with_cooldown(group_id, now)
                return False, "without human limit"

        if bool(limits.get("enable_duplicate_message_limit", True)):
            window = self._limit_int("duplicate_window_seconds", 300, minimum=1)
            max_same = self._limit_int("max_same_message_per_session", 1, minimum=1)
            msg_hash = _stable_message_hash(event.message_str)
            state.recent_messages = self._prune_recent_messages(state.recent_messages, now, window)
            same_count = sum(1 for _, item_hash in state.recent_messages if item_hash == msg_hash)
            if same_count >= max_same:
                return False, "duplicate message limit"

        return True, "ok"

    def _prune_recent_messages(
        self,
        values: list[tuple[float, str]],
        now: float,
        window: int,
    ) -> list[tuple[float, str]]:
        if window <= 0:
            return []
        return [(ts, msg_hash) for ts, msg_hash in values if now - ts <= window]

    def _record_bot_request(
        self,
        event: AstrMessageEvent,
        state: BotSessionState,
        now: float,
    ) -> None:
        group_id = _normalize_id(event.get_group_id()) or "nogroup"
        state.turns += 1
        state.last_accepted_at = now
        state.recent_messages.append((now, _stable_message_hash(event.message_str)))
        self.group_window_times.setdefault(group_id, []).append(now)
        self.no_human_times.setdefault(group_id, []).append(now)

    def _self_platform_nickname(self, event: AstrMessageEvent, self_entry: BotEntry) -> str:
        self_id = _normalize_id(event.get_self_id())
        group = getattr(event.message_obj, "group", None)
        members = getattr(group, "members", None) if group else None
        if members:
            for member in members:
                if _normalize_id(getattr(member, "user_id", "")) == self_id:
                    nickname = str(getattr(member, "nickname", "") or "").strip()
                    if nickname:
                        return nickname
        return self_entry.display_name

    def _bot_prompt(self, event: AstrMessageEvent, peer: BotEntry, self_entry: BotEntry) -> str:
        prompting = self._section("prompting")
        template = str(prompting.get("bot_prompt_template") or "")
        style = str(prompting.get("reply_style_prompt") or "")
        sender_name = (event.get_sender_name() or "").strip() or "未知"
        self_nickname = self._self_platform_nickname(event, self_entry)
        replacements = {
            "<botname>": peer.display_name,
            "<callname>": peer.effective_call_name,
            "<botid>": peer.qq or "未知",
            "<bottype>": "受控机器人" if peer.kind == "controlled" else "不受控机器人",
            "<selfname>": self_entry.display_name,
            "<selfnickname>": self_nickname,
            "<sendername>": sender_name,
            "<configname>": peer.name or "未知",
        }
        text = f"{template}\n\n{style}".strip()
        for key, value in replacements.items():
            text = text.replace(key, value)
        return text

    def _set_bot_context_extra(
        self,
        event: AstrMessageEvent,
        peer: BotEntry,
        self_entry: BotEntry,
        session_key: str,
    ) -> None:
        event.set_extra("multi_bot_control_peer", peer.to_json())
        event.set_extra("multi_bot_control_session_key", session_key)
        event.set_extra("multi_bot_control_prompt", self._bot_prompt(event, peer, self_entry))

    async def _delay_before_llm(self, base_delay: float, rank_delay: float) -> None:
        delay = max(0.0, base_delay + rank_delay)
        if delay > 0:
            await asyncio.sleep(delay)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=100)
    async def route_group_messages(self, event: AstrMessageEvent):
        """多机器人群消息前置路由。"""
        if not self._enabled():
            return

        group_id = _normalize_id(event.get_group_id())
        if not group_id:
            return

        sender_id = _normalize_id(event.get_sender_id())
        self_id = _normalize_id(event.get_self_id())
        if sender_id and self_id and sender_id == self_id:
            event.stop_event()
            return

        if not event.is_at_or_wake_command:
            return

        entries = self._effective_entries(group_id)
        self_entry = self._self_entry(event, entries)
        targeted_controlled = self._targeted_controlled_entries(event, entries)

        peer = self._find_peer_bot(event, entries)
        if not peer:
            self._reset_group_turns(group_id)
            if event.is_at_or_wake_command and len(targeted_controlled) > 1:
                rank_delay = self._self_rank_delay(self_entry, targeted_controlled)
                await self._delay_before_llm(0.0, rank_delay)
            return

        if peer.kind == "uncontrolled" and peer.source == "global":
            logger.info(f"外部机器人 {self._bot_log_id(peer)} 发言")

        self._clear_activated_handlers(event)

        if targeted_controlled and self_entry.identity not in {entry.identity for entry in targeted_controlled}:
            self._log_controlled_reply_blocked(self_entry, self._state_for(self._session_key(event, peer)), "targeted other controlled bot", time.time())
            event.stop_event()
            return

        now = time.time()
        session_key = self._session_key(event, peer)
        state = self._state_for(session_key)
        ok, reason = self._can_accept_bot_request(event, peer, state, now)
        if not ok:
            logger.info(f"{PLUGIN_NAME}: 静默拦截机器人消息: {reason}")
            self._log_group_reply_blocked(event, self_entry, state, reason, now)
            event.stop_event()
            return

        self._record_bot_request(event, state, now)
        if self_entry.kind == "controlled" and self_entry.source == "global":
            remaining_turns = self._remaining_session_turns(state)
            logger.info(
                f"受控机器人 {self._bot_log_id(self_entry)} 回复了 bot 的发言，剩余回复轮次：{remaining_turns}",
            )
        self._set_bot_context_extra(event, peer, self_entry, session_key)
        event.is_wake = True
        event.is_at_or_wake_command = True

        base_delay = float(self._limit_int("bot_request_delay_seconds", 2))
        rank_delay = self._self_rank_delay(self_entry, targeted_controlled)
        await self._delay_before_llm(base_delay, rank_delay)

    def _clear_activated_handlers(self, event: AstrMessageEvent) -> None:
        activated_handlers = event.get_extra("activated_handlers", [])
        if isinstance(activated_handlers, list):
            activated_handlers.clear()
        else:
            event.set_extra("activated_handlers", [])
        event.set_extra("handlers_parsed_params", {})

    @filter.on_llm_request(priority=100)
    async def inject_bot_prompt(self, event: AstrMessageEvent, req: ProviderRequest):
        """为机器人来源消息注入临时称呼和回复风格提示。"""
        if not self._enabled():
            return
        prompt = event.get_extra("multi_bot_control_prompt")
        if not prompt:
            if self._is_unapproved_bot_llm_request(event):
                logger.info(f"{PLUGIN_NAME}: 在 LLM 请求阶段兜底拦截未放行的机器人消息。")
                event.stop_event()
            return
        if TextPart is not None and hasattr(req, "extra_user_content_parts"):
            try:
                part = TextPart(text=prompt)
                mark_as_temp = getattr(part, "mark_as_temp", None)
                if callable(mark_as_temp):
                    part = mark_as_temp()
                elif hasattr(part, "_no_save"):
                    try:
                        part._no_save = True
                    except Exception:
                        pass
                req.extra_user_content_parts.append(part)
                return
            except Exception as e:
                logger.warning(f"{PLUGIN_NAME}: 注入 extra_user_content_parts 失败，回退到 system_prompt: {e}")
        req.system_prompt = f"{req.system_prompt}\n\n{prompt}".strip()

    def _is_unapproved_bot_llm_request(self, event: AstrMessageEvent) -> bool:
        if not event.is_at_or_wake_command:
            return False
        group_id = _normalize_id(event.get_group_id())
        if not group_id:
            return False
        entries = self._effective_entries(group_id)
        peer = self._find_peer_bot(event, entries)
        if not peer:
            return False
        sender_id = _normalize_id(event.get_sender_id())
        self_id = _normalize_id(event.get_self_id())
        return bool(sender_id and sender_id != self_id)

    @filter.command_group("mbot", alias={"多机器人", "botctl"})
    def mbot(self):
        pass

    @mbot.command("help", alias={"帮助"})
    async def mbot_help(self, event: AstrMessageEvent):
        """查看多机器人本群名单命令帮助。"""
        yield event.plain_result(self._mbot_help_text())
        event.stop_event()

    @filter.regex(r"^/(?:mbot|多机器人|botctl)(?:\s|$)", priority=90)
    async def mbot_slash_fallback(self, event: AstrMessageEvent):
        """允许 QQ 群主直接使用 /mbot 命令管理当前群名单。"""
        async for result in self._dispatch_slash_mbot(event):
            yield result
        event.stop_event()

    @mbot.command("list", alias={"ls", "列表"})
    async def mbot_list(self, event: AstrMessageEvent):
        """查看当前群本地机器人名单。"""
        group_id = _normalize_id(event.get_group_id())
        if not group_id:
            yield event.plain_result("该命令只能在群聊中使用。")
            event.stop_event()
            return
        entries = self._local_entries(group_id)
        if not entries:
            yield event.plain_result("当前群还没有由群主设置的本地机器人名单。")
            event.stop_event()
            return
        lines = ["当前群本地机器人名单："]
        for idx, entry in enumerate(entries, start=1):
            lines.append(
                f"{idx}. QQ={entry.qq} 昵称={entry.name or '未设置'} 称呼={entry.effective_call_name}",
            )
        yield event.plain_result("\n".join(lines))
        event.stop_event()

    @mbot.command("add", alias={"添加"})
    async def mbot_add(
        self,
        event: AstrMessageEvent,
        qq: str,
        name: str = "",
        call_name: str = "",
    ):
        """由 QQ 群群主添加当前群本地机器人名单。"""
        ok, message = await self._ensure_group_owner(event)
        if not ok:
            yield event.plain_result(message)
            event.stop_event()
            return
        group_id = _normalize_id(event.get_group_id())
        entry = BotEntry(
            qq=_normalize_id(qq),
            name=name.strip(),
            call_name=call_name.strip(),
            kind="uncontrolled",
            source="group",
            allow_interaction=True,
        )
        if not entry.qq:
            yield event.plain_result("QQ号不能为空，机器人识别仅支持 QQ 号精确匹配。")
            event.stop_event()
            return
        self._upsert_local_entry(group_id, entry)
        yield event.plain_result(f"已添加/更新当前群本地机器人：{entry.display_name}。")
        event.stop_event()

    @mbot.command("remove", alias={"rm", "删除"})
    async def mbot_remove(self, event: AstrMessageEvent, key: str):
        """由 QQ 群群主删除当前群本地机器人名单。"""
        ok, message = await self._ensure_group_owner(event)
        if not ok:
            yield event.plain_result(message)
            event.stop_event()
            return
        group_id = _normalize_id(event.get_group_id())
        removed = self._remove_local_entry(group_id, key)
        if removed:
            yield event.plain_result(f"已删除当前群本地机器人：{removed.display_name}。")
        else:
            yield event.plain_result("未找到匹配的当前群本地机器人。")
        event.stop_event()

    @mbot.command("clear", alias={"清空"})
    async def mbot_clear(self, event: AstrMessageEvent):
        """由 QQ 群群主清空当前群本地机器人名单。"""
        ok, message = await self._ensure_group_owner(event)
        if not ok:
            yield event.plain_result(message)
            event.stop_event()
            return
        group_id = _normalize_id(event.get_group_id())
        data = self._load_group_bots_data()
        data.setdefault("groups", {}).pop(group_id, None)
        self._save_group_bots_data()
        yield event.plain_result("已清空当前群本地机器人名单。")
        event.stop_event()

    def _upsert_local_entry(self, group_id: str, entry: BotEntry) -> None:
        data = self._load_group_bots_data()
        group = data.setdefault("groups", {}).setdefault(group_id, {"bots": []})
        bots = group.setdefault("bots", [])
        key = entry.qq.casefold()
        replaced = False
        for idx, raw in enumerate(list(bots)):
            old = self._normalize_bot_entry(raw, "uncontrolled", "group")
            if not old:
                continue
            old_key = old.qq.casefold()
            if old_key == key:
                bots[idx] = entry.to_json()
                replaced = True
                break
        if not replaced:
            bots.append(entry.to_json())
        group["updated_at"] = int(time.time())
        self._save_group_bots_data()

    def _remove_local_entry(self, group_id: str, key: str) -> BotEntry | None:
        data = self._load_group_bots_data()
        group = data.setdefault("groups", {}).setdefault(group_id, {"bots": []})
        bots = group.setdefault("bots", [])
        key_fold = key.strip().casefold()
        for idx, raw in enumerate(list(bots)):
            entry = self._normalize_bot_entry(raw, "uncontrolled", "group")
            if not entry:
                continue
            if entry.qq.casefold() == key_fold:
                bots.pop(idx)
                group["updated_at"] = int(time.time())
                self._save_group_bots_data()
                return entry
        return None

    async def _dispatch_slash_mbot(self, event: AstrMessageEvent):
        try:
            parts = shlex.split(event.message_str.strip())
        except ValueError as e:
            yield event.plain_result(f"命令解析失败：{e}")
            return
        if not parts:
            yield event.plain_result(self._mbot_help_text())
            return
        if parts[0].startswith("/"):
            parts[0] = parts[0][1:]
        if not parts or parts[0] not in {"mbot", "多机器人", "botctl"}:
            return
        cmd = parts[1] if len(parts) > 1 else "help"
        args = parts[2:]

        if cmd in {"help", "帮助"}:
            yield event.plain_result(self._mbot_help_text())
            return
        if cmd in {"list", "ls", "列表"}:
            async for item in self.mbot_list(event):
                yield item
            return
        if cmd in {"clear", "清空"}:
            async for item in self.mbot_clear(event):
                yield item
            return
        if cmd in {"remove", "rm", "删除"}:
            if not args:
                yield event.plain_result("用法：/mbot remove <QQ号>")
                return
            async for item in self.mbot_remove(event, args[0]):
                yield item
            return
        if cmd in {"add", "添加"}:
            if not args:
                yield event.plain_result(
                    "用法：/mbot add <QQ号> [昵称] [称呼]",
                )
                return
            padded = [*args, "", ""][:3]
            async for item in self.mbot_add(event, padded[0], padded[1], padded[2]):
                yield item
            return
        yield event.plain_result(self._mbot_help_text())

    def _mbot_help_text(self) -> str:
        return (
            "多机器人本群名单命令：\n"
            "/mbot add <QQ号> [昵称] [称呼]\n"
            "/mbot remove <QQ号>\n"
            "/mbot list\n"
            "/mbot clear\n"
            "说明：add/remove/clear 仅 QQ 群群主可用；这些命令只修改当前群的本地名单，不会修改 WebUI 全局配置和交互限制。机器人识别仅支持 QQ 号精确匹配。"
        )

    async def _ensure_group_owner(self, event: AstrMessageEvent) -> tuple[bool, str]:
        if not event.get_group_id():
            return False, "该命令只能在群聊中使用。"
        platform_name = (event.get_platform_name() or "").casefold()
        if "qq" not in platform_name and "aiocqhttp" not in platform_name:
            return False, "该命令只允许 QQ 群群主使用。"
        sender_id = _normalize_id(event.get_sender_id())
        if self._raw_sender_is_owner(event):
            return True, ""
        owner_id = self._group_owner_from_event(event)
        if not owner_id:
            try:
                group = await event.get_group()
                owner_id = _normalize_id(getattr(group, "group_owner", "")) if group else ""
            except Exception as e:
                logger.warning(f"{PLUGIN_NAME}: 获取群信息失败: {e}")
        if owner_id and sender_id == owner_id:
            return True, ""
        if owner_id:
            return False, "只有 QQ 群群主可以修改当前群本地机器人名单。"
        return False, "无法确认群主身份，已拒绝修改当前群本地机器人名单。"

    def _group_owner_from_event(self, event: AstrMessageEvent) -> str:
        group = getattr(event.message_obj, "group", None)
        return _normalize_id(getattr(group, "group_owner", "")) if group else ""

    def _raw_sender_is_owner(self, event: AstrMessageEvent) -> bool:
        raw = getattr(event.message_obj, "raw_message", None)
        role = ""
        if isinstance(raw, dict):
            sender = raw.get("sender") or {}
            if isinstance(sender, dict):
                role = str(sender.get("role") or "")
            role = role or str(raw.get("role") or "")
        else:
            sender = getattr(raw, "sender", None)
            role = str(getattr(sender, "role", "") or getattr(raw, "role", ""))
        return role.casefold() == "owner"

    async def terminate(self):
        self._save_group_bots_data()
        self._save_priorities()
