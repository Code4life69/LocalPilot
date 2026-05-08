from __future__ import annotations

import json
import queue
import sys
import threading
import tkinter as tk
import atexit
from pathlib import Path
from tkinter import messagebox, scrolledtext
from typing import Any

from app.git_sync import GitSyncManager
from app.llm.ollama_client import OllamaClient
from app.llm.prompts import build_system_prompt
from app.logger import AppLogger
from app.memory import MemoryStore
from app.modes.chat_mode import ChatMode
from app.modes.code_mode import CodeMode
from app.modes.desktop_mode import DesktopMode
from app.modes.research_mode import ResearchMode
from app.router import KeywordRouter
from app.safety import SafetyManager


class LocalPilotApp:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.settings = self._load_json(self.root_dir / "config" / "settings.json")
        self.model_profiles = self._load_json(self.root_dir / "config" / "model_profiles.json")
        self.logger = AppLogger(self.root_dir / self.settings["logs_dir"])
        self.git_sync = GitSyncManager(self.root_dir, self.settings, self.logger)
        self.memory = MemoryStore(
            self.root_dir / self.settings["memory_dir"],
            self.root_dir / "config" / "capabilities.json",
        )
        self.capabilities = self.memory.load_capabilities()
        self.system_prompt = build_system_prompt(self.capabilities)
        self.router = KeywordRouter()
        self.ollama = OllamaClient(
            host=self.model_profiles["ollama"]["host"],
            timeout_seconds=self.model_profiles["ollama"]["timeout_seconds"],
            main_model=self.model_profiles["models"]["main"],
            vision_model=self.model_profiles["models"]["vision"],
        )
        self._initialize_ollama()
        self.safety = SafetyManager(approval_callback=self._approval_callback)
        self.gui: LocalPilotGUI | None = None
        self._shutdown_complete = False
        self.modes = {
            "chat": ChatMode(self),
            "code": CodeMode(self),
            "research": ResearchMode(self),
            "desktop": DesktopMode(self),
        }
        self._run_git_sync("startup")

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def attach_gui(self, gui: "LocalPilotGUI") -> None:
        self.gui = gui
        self.logger.register_callback(gui.on_event)

    def _initialize_ollama(self) -> None:
        ollama_settings = self.settings.get("ollama", {})
        ok, message = self.ollama.ensure_server(
            auto_start=bool(ollama_settings.get("auto_start_server", True)),
            wait_seconds=int(ollama_settings.get("startup_wait_seconds", 8)),
        )
        role = "Reasoner" if ok else "Ollama"
        self.logger.event(role, message)

    def _run_git_sync(self, trigger: str) -> None:
        ok, message = self.git_sync.sync(trigger)
        role = "GitSync" if ok else "GitSyncWarning"
        self.logger.event(role, message, persist=False, trigger=trigger)

    def _approval_callback(self, prompt: str) -> bool:
        self.logger.event("Safety", "Confirmation required", prompt=prompt)
        if self.gui is not None:
            return self.gui.request_approval(prompt)
        return self._cli_approval(prompt)

    def _cli_approval(self, prompt: str) -> bool:
        reply = input(f"{prompt}\nApprove? y/n: ").strip().lower()
        return reply == "y"

    def ask_approval(self, prompt: str) -> bool:
        return self.safety.confirm(prompt)

    def describe_capabilities(self) -> str:
        caps = self.capabilities
        return (
            f"{caps['name']}: {caps['purpose']}\n"
            f"Modes: {', '.join(caps['modes'])}\n"
            f"Safety: {'; '.join(caps['safety_rules'])}\n"
            f"Known limits: {'; '.join(caps['known_limits'])}"
        )

    def process_user_input(self, user_text: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "user_text": user_text,
            "mode": self.router.classify(user_text),
            "requires_confirmation": False,
            "approved": None,
            "result": None,
            "events": [],
        }
        self.logger.event("Router", f"classified as {request['mode']}", user_text=user_text)
        request["events"].append({"role": "Router", "message": f"classified as {request['mode']}"})
        self.logger.event("Reasoner", f"dispatching mode {request['mode']}")
        self.logger.event(f"Mode:{request['mode']}", "activated")

        if request["mode"] == "memory":
            result = self._handle_memory_request(request)
        else:
            handler = self.modes.get(request["mode"], self.modes["chat"])
            result = handler.handle(request)
        request["result"] = result
        return request

    def _handle_memory_request(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request["user_text"].strip()
        lowered = text.lower()
        self.logger.event("Memory", f"Handling memory request: {text}")

        if lowered.startswith("save note") or lowered.startswith("remember"):
            note_text = text.split(" ", 2)[-1] if " " in text else ""
            return {"ok": True, "message": self.memory.save_note(note_text)}

        if lowered.startswith("search notes"):
            keyword = text.split(" ", 2)[-1] if " " in text else ""
            matches = self.memory.search_notes(keyword)
            return {"ok": True, "matches": matches}

        if lowered.startswith("show notes") or lowered == "notes":
            return {"ok": True, "content": self.memory.show_notes()}

        if lowered.startswith("save fact"):
            parts = text.split(" ", 3)
            if len(parts) < 4:
                return {"ok": False, "error": "Use: save fact <key> <value>"}
            return {"ok": True, "message": self.memory.save_fact(parts[2], parts[3])}

        return {
            "ok": True,
            "message": (
                "Memory mode supports: save note ..., search notes ..., show notes, save fact <key> <value>."
            ),
        }

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self._run_git_sync("shutdown")


class LocalPilotGUI:
    def __init__(self, app: LocalPilotApp) -> None:
        self.app = app
        self.root = tk.Tk()
        self.root.title("LocalPilot")
        self.root.geometry("1100x720")
        self.event_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._build_widgets()
        self.root.after(150, self._drain_events)

    def _build_widgets(self) -> None:
        header = tk.Frame(self.root)
        header.pack(fill="x", padx=8, pady=8)

        self.mode_var = tk.StringVar(value="Mode: idle")
        self.role_var = tk.StringVar(value="Role: idle")
        tk.Label(header, textvariable=self.mode_var, font=("Segoe UI", 11, "bold")).pack(side="left", padx=8)
        tk.Label(header, textvariable=self.role_var, font=("Segoe UI", 11, "bold")).pack(side="left", padx=8)

        body = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = tk.Frame(body)
        right = tk.Frame(body)
        body.add(left, stretch="always")
        body.add(right)

        tk.Label(left, text="Conversation / Output", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.output = scrolledtext.ScrolledText(left, wrap=tk.WORD, height=25)
        self.output.pack(fill="both", expand=True)
        self.output.configure(state="disabled")

        input_frame = tk.Frame(left)
        input_frame.pack(fill="x", pady=(8, 0))
        self.input_entry = tk.Entry(input_frame)
        self.input_entry.pack(side="left", fill="x", expand=True)
        self.input_entry.bind("<Return>", lambda _event: self.submit_input())
        tk.Button(input_frame, text="Send", command=self.submit_input).pack(side="left", padx=(8, 0))

        tk.Label(right, text="Activity Timeline", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.timeline = scrolledtext.ScrolledText(right, wrap=tk.WORD, width=45, height=20)
        self.timeline.pack(fill="both", expand=True)
        self.timeline.configure(state="disabled")

        tk.Label(right, text="Recent Logs", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.logs = scrolledtext.ScrolledText(right, wrap=tk.WORD, width=45, height=12)
        self.logs.pack(fill="both", expand=True)
        self.logs.configure(state="disabled")

        for widget in (self.output, self.timeline, self.logs):
            widget.bind("<Key>", lambda _event: "break")
            widget.bind("<<Paste>>", lambda _event: "break")
            widget.bind("<Button-3>", lambda _event: "break")

    def submit_input(self) -> None:
        text = self.input_entry.get().strip()
        if not text:
            return
        self.input_entry.delete(0, tk.END)
        self._append_readonly(self.output, f"\nYou: {text}\n")
        request = self.app.process_user_input(text)
        self._append_readonly(self.output, f"LocalPilot: {format_result(request['result'])}\n")

    def on_event(self, event: dict[str, Any]) -> None:
        self.event_queue.put(event)

    def _drain_events(self) -> None:
        while not self.event_queue.empty():
            event = self.event_queue.get()
            role = event["role"]
            message = event["message"]
            self.role_var.set(f"Role: {role}")
            if role.startswith("Mode:"):
                self.mode_var.set(role.replace("Mode:", "Mode: "))
            line = f"[{event['timestamp']}] {role} -> {message}\n"
            self._append_readonly(self.timeline, line)
            self._append_readonly(self.logs, line)
        self.root.after(150, self._drain_events)

    def request_approval(self, prompt: str) -> bool:
        approved = {"value": False}
        done = threading.Event()

        def ask() -> None:
            approved["value"] = messagebox.askyesno("LocalPilot Approval", prompt, parent=self.root)
            done.set()

        self.root.after(0, ask)
        done.wait()
        return approved["value"]

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self) -> None:
        self.app.shutdown()
        self.root.destroy()

    def _append_readonly(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state="disabled")


def format_result(result: dict[str, Any]) -> str:
    if "message" in result:
        return str(result["message"])
    if result.get("results"):
        lines = [f"Research results for: {result.get('query', '')}"]
        for item in result["results"]:
            lines.append(f"- {item.get('title', '')}")
            lines.append(f"  {item.get('url', '')}")
            if item.get("snippet"):
                lines.append(f"  {item['snippet']}")
        return "\n".join(lines)
    return json.dumps(result, indent=2)


def safe_console_print(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sanitized = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(sanitized)


def run_cli(app: LocalPilotApp) -> None:
    safe_console_print("LocalPilot CLI started. Type 'exit' to quit.")
    safe_console_print(app.describe_capabilities())
    if app.ollama.last_status not in {"running", "started_by_localpilot"}:
        safe_console_print()
        safe_console_print(
            app.ollama.build_unavailable_message(auto_start_attempted=app.ollama.last_status == "start_timeout")
        )
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        request = app.process_user_input(user_text)
        safe_console_print(f"\nLocalPilot> {format_result(request['result'])}")


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    app = LocalPilotApp(root_dir)
    atexit.register(app.shutdown)
    enable_gui = bool(app.settings.get("enable_gui", True))

    if enable_gui:
        try:
            gui = LocalPilotGUI(app)
            app.attach_gui(gui)
            cli_thread = threading.Thread(target=run_cli, args=(app,), daemon=True)
            cli_thread.start()
            gui.run()
            return 0
        except Exception as exc:
            app.logger.event("GUI", f"GUI unavailable, falling back to CLI: {exc}")

    run_cli(app)
    return 0
