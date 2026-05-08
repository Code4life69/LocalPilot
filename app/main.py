from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext
from typing import Any

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
        self.safety = SafetyManager(approval_callback=self._approval_callback)
        self.gui: LocalPilotGUI | None = None
        self.modes = {
            "chat": ChatMode(self),
            "code": CodeMode(self),
            "research": ResearchMode(self),
            "desktop": DesktopMode(self),
        }

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def attach_gui(self, gui: "LocalPilotGUI") -> None:
        self.gui = gui
        self.logger.register_callback(gui.on_event)

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

        input_frame = tk.Frame(left)
        input_frame.pack(fill="x", pady=(8, 0))
        self.input_entry = tk.Entry(input_frame)
        self.input_entry.pack(side="left", fill="x", expand=True)
        self.input_entry.bind("<Return>", lambda _event: self.submit_input())
        tk.Button(input_frame, text="Send", command=self.submit_input).pack(side="left", padx=(8, 0))

        tk.Label(right, text="Activity Timeline", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.timeline = scrolledtext.ScrolledText(right, wrap=tk.WORD, width=45, height=20)
        self.timeline.pack(fill="both", expand=True)

        tk.Label(right, text="Recent Logs", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.logs = scrolledtext.ScrolledText(right, wrap=tk.WORD, width=45, height=12)
        self.logs.pack(fill="both", expand=True)

    def submit_input(self) -> None:
        text = self.input_entry.get().strip()
        if not text:
            return
        self.input_entry.delete(0, tk.END)
        self.output.insert(tk.END, f"\nYou: {text}\n")
        request = self.app.process_user_input(text)
        self.output.insert(tk.END, f"LocalPilot: {format_result(request['result'])}\n")
        self.output.see(tk.END)

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
            self.timeline.insert(tk.END, line)
            self.timeline.see(tk.END)
            self.logs.insert(tk.END, line)
            self.logs.see(tk.END)
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
        self.root.mainloop()


def format_result(result: dict[str, Any]) -> str:
    if "message" in result:
        return str(result["message"])
    return json.dumps(result, indent=2)


def run_cli(app: LocalPilotApp) -> None:
    print("LocalPilot CLI started. Type 'exit' to quit.")
    print(app.describe_capabilities())
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
        print(f"\nLocalPilot> {format_result(request['result'])}")


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    app = LocalPilotApp(root_dir)
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
