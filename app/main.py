from __future__ import annotations

import json
import queue
import sys
import threading
import tkinter as tk
import atexit
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
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

    def run_guarded_desktop_action(self, action_name: str, action):
        self.logger.event("DesktopGuard", f"starting {action_name}")
        if self.gui is not None:
            self.gui.show_desktop_busy_overlay(action_name)
        try:
            return action()
        finally:
            if self.gui is not None:
                self.gui.hide_desktop_busy_overlay()
            self.logger.event("DesktopGuard", f"finished {action_name}")


class LocalPilotGUI:
    def __init__(self, app: LocalPilotApp) -> None:
        self.app = app
        self.root = tk.Tk()
        self.root.title("LocalPilot")
        self.root.geometry("1240x820")
        self.root.minsize(1100, 760)
        self.event_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.desktop_overlay: tk.Toplevel | None = None
        self.memory_text: scrolledtext.ScrolledText | None = None
        self._build_widgets()
        self._refresh_status_bar()
        self.root.after(150, self._drain_events)

    def _build_widgets(self) -> None:
        theme = self.app.settings.get("ui", {}).get("theme", "dark")
        colors = self._theme_colors(theme)
        self.colors = colors
        self.root.configure(bg=colors["bg"])
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Status.TFrame", background=colors["panel"])
        self.style.configure("StatusLabel.TLabel", background=colors["panel"], foreground=colors["text"], font=("Segoe UI", 10))
        self.style.configure("StatusValue.TLabel", background=colors["panel"], foreground=colors["accent"], font=("Segoe UI", 10, "bold"))
        self.style.configure("Tabs.TNotebook", background=colors["bg"], borderwidth=0)
        self.style.configure("Tabs.TNotebook.Tab", font=("Segoe UI", 10), padding=(14, 8), background=colors["panel"], foreground=colors["muted"])
        self.style.map("Tabs.TNotebook.Tab", background=[("selected", colors["surface"])], foreground=[("selected", colors["text"])])
        self.style.configure("Action.TButton", font=("Segoe UI", 10), padding=(10, 8))

        header = ttk.Frame(self.root, style="Status.TFrame", padding=(14, 12))
        header.pack(fill="x", padx=14, pady=(14, 10))

        self.mode_var = tk.StringVar(value="idle")
        self.role_var = tk.StringVar(value="idle")
        self.ollama_var = tk.StringVar(value="unknown")
        self.main_model_var = tk.StringVar(value="n/a")
        self.vision_model_var = tk.StringVar(value="n/a")
        self.safety_var = tk.StringVar(value="Guarded")

        status_items = [
            ("Mode", self.mode_var),
            ("Role", self.role_var),
            ("Ollama", self.ollama_var),
            ("Main", self.main_model_var),
            ("Vision", self.vision_model_var),
            ("Safety", self.safety_var),
        ]
        for index, (label, variable) in enumerate(status_items):
            item = ttk.Frame(header, style="Status.TFrame")
            item.grid(row=0, column=index, sticky="w", padx=(0, 18))
            ttk.Label(item, text=f"{label}:", style="StatusLabel.TLabel").pack(side="left")
            ttk.Label(item, textvariable=variable, style="StatusValue.TLabel").pack(side="left", padx=(6, 0))

        body = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            bg=colors["bg"],
            sashwidth=8,
            bd=0,
            highlightthickness=0,
        )
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        left = tk.Frame(body, bg=colors["bg"])
        right = tk.Frame(body, bg=colors["bg"])
        body.add(left, stretch="always", minsize=720)
        body.add(right, minsize=360)

        tk.Label(
            left,
            text="Conversation",
            font=("Segoe UI", 11, "bold"),
            bg=colors["bg"],
            fg=colors["text"],
        ).pack(anchor="w", pady=(0, 8))
        self.output = scrolledtext.ScrolledText(
            left,
            wrap=tk.WORD,
            height=25,
            font=("Segoe UI", 11),
            bg=colors["surface"],
            fg=colors["text"],
            insertbackground=colors["text"],
            relief="flat",
            bd=0,
            padx=14,
            pady=14,
            spacing1=6,
            spacing3=10,
        )
        self.output.pack(fill="both", expand=True)
        self.output.configure(state="disabled")
        self.output.tag_configure("user", font=("Segoe UI", 11, "bold"), foreground=colors["accent"])
        self.output.tag_configure("assistant", font=("Segoe UI", 11, "bold"), foreground=colors["success"])
        self.output.tag_configure("body", font=("Segoe UI", 11), foreground=colors["text"])
        self.output.tag_configure("error", font=("Segoe UI", 11), foreground=colors["danger"])

        input_frame = tk.Frame(left, bg=colors["bg"])
        input_frame.pack(fill="x", pady=(12, 0))
        self.input_entry = tk.Entry(
            input_frame,
            font=("Segoe UI", 11),
            bg=colors["surface"],
            fg=colors["text"],
            insertbackground=colors["text"],
            relief="flat",
            bd=0,
        )
        self.input_entry.pack(side="left", fill="x", expand=True)
        self.input_entry.bind("<Return>", lambda _event: self.submit_input())
        send_button = ttk.Button(input_frame, text="Send", command=self.submit_input, style="Action.TButton")
        send_button.pack(side="left", padx=(10, 0))

        notebook = ttk.Notebook(right, style="Tabs.TNotebook")
        notebook.pack(fill="both", expand=True)

        activity_tab = tk.Frame(notebook, bg=colors["bg"])
        logs_tab = tk.Frame(notebook, bg=colors["bg"])
        memory_tab = tk.Frame(notebook, bg=colors["bg"])
        tools_tab = tk.Frame(notebook, bg=colors["bg"])
        notebook.add(activity_tab, text="Activity")
        notebook.add(logs_tab, text="Logs")
        notebook.add(memory_tab, text="Memory")
        notebook.add(tools_tab, text="Tools")

        self.timeline = self._make_panel_text(activity_tab, height=28)
        self.logs = self._make_panel_text(logs_tab, height=28)
        self.memory_text = self._make_panel_text(memory_tab, height=28)
        self._load_memory_panel()
        self._build_tools_tab(tools_tab)

        for widget in (self.output, self.timeline, self.logs, self.memory_text):
            widget.bind("<Key>", lambda _event: "break")
            widget.bind("<<Paste>>", lambda _event: "break")
            widget.bind("<Button-3>", lambda _event: "break")

    def submit_input(self) -> None:
        text = self.input_entry.get().strip()
        if not text:
            return
        self.input_entry.delete(0, tk.END)
        self.submit_text(text)

    def submit_text(self, text: str) -> None:
        self._append_chat_message("You", text, speaker_tag="user")
        request = self.app.process_user_input(text)
        rendered = format_result(request["result"])
        speaker_tag = "assistant"
        body_tag = "body"
        if isinstance(request["result"], dict) and request["result"].get("error"):
            body_tag = "error"
        self._append_chat_message("LocalPilot", rendered, speaker_tag=speaker_tag, body_tag=body_tag)
        self._refresh_status_bar()
        self._maybe_refresh_memory(request["result"])

    def on_event(self, event: dict[str, Any]) -> None:
        self.event_queue.put(event)

    def _drain_events(self) -> None:
        while not self.event_queue.empty():
            event = self.event_queue.get()
            role = event["role"]
            message = event["message"]
            self.role_var.set(role)
            if role.startswith("Mode:"):
                self.mode_var.set(role.replace("Mode:", "").strip())
            line = f"[{event['timestamp']}] {role} -> {message}\n"
            self._append_readonly(self.timeline, line)
            self._append_readonly(self.logs, line)
            self._refresh_status_bar()
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

    def _append_chat_message(self, speaker: str, text: str, speaker_tag: str, body_tag: str = "body") -> None:
        widget = self.output
        widget.configure(state="normal")
        widget.insert(tk.END, f"{speaker}\n", (speaker_tag,))
        widget.insert(tk.END, f"{text}\n\n", (body_tag,))
        widget.see(tk.END)
        widget.configure(state="disabled")

    def _make_panel_text(self, parent: tk.Frame, height: int) -> scrolledtext.ScrolledText:
        text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            height=height,
            font=("Segoe UI", 10),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
        )
        text.pack(fill="both", expand=True, padx=2, pady=2)
        text.configure(state="disabled")
        return text

    def _build_tools_tab(self, parent: tk.Frame) -> None:
        container = tk.Frame(parent, bg=self.colors["bg"])
        container.pack(fill="both", expand=True, padx=8, pady=8)

        actions = [
            ("Show Notes", lambda: self.submit_text("show notes")),
            ("Take Screenshot", lambda: self.submit_text("take screenshot")),
            ("Mouse Position", lambda: self.submit_text("get mouse position")),
            ("Clear Chat", self.clear_chat),
        ]
        for label, command in actions:
            button = ttk.Button(container, text=label, command=command, style="Action.TButton")
            button.pack(fill="x", pady=6)

    def clear_chat(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.configure(state="disabled")

    def _refresh_status_bar(self) -> None:
        self.ollama_var.set(self.app.ollama.last_status.replace("_", " "))
        self.main_model_var.set(self.app.ollama.active_main_model or self.app.model_profiles["models"]["main"])
        self.vision_model_var.set(self.app.ollama.active_vision_model or self.app.model_profiles["models"]["vision"])
        self.safety_var.set("Guarded")

    def _load_memory_panel(self) -> None:
        if self.memory_text is None:
            return
        content = self.app.memory.show_notes()
        self.memory_text.configure(state="normal")
        self.memory_text.delete("1.0", tk.END)
        self.memory_text.insert(tk.END, content)
        self.memory_text.configure(state="disabled")

    def _maybe_refresh_memory(self, result: dict[str, Any]) -> None:
        if any(key in result for key in ("content", "matches", "message")):
            self._load_memory_panel()

    def _theme_colors(self, theme: str) -> dict[str, str]:
        if theme == "light":
            return {
                "bg": "#edf1f5",
                "panel": "#dce5ef",
                "surface": "#ffffff",
                "text": "#16212e",
                "muted": "#5b6774",
                "accent": "#185ea8",
                "success": "#18794e",
                "danger": "#b42318",
            }
        return {
            "bg": "#0f141b",
            "panel": "#18202a",
            "surface": "#1f2935",
            "text": "#f3f6fb",
            "muted": "#a2afbf",
            "accent": "#61b0ff",
            "success": "#6fd3a5",
            "danger": "#ff8f8f",
        }

    def show_desktop_busy_overlay(self, action_name: str) -> None:
        settings = self.app.settings.get("desktop_guard", {})
        if not settings.get("show_overlay", True):
            return

        def build_overlay() -> None:
            if self.desktop_overlay is not None and self.desktop_overlay.winfo_exists():
                return

            overlay = tk.Toplevel(self.root)
            overlay.title(settings.get("title", "LocalPilot Is Using Your PC"))
            overlay.attributes("-topmost", True)
            overlay.geometry("560x220+480+220")
            overlay.configure(bg="#101820")
            overlay.resizable(False, False)
            overlay.protocol("WM_DELETE_WINDOW", lambda: None)
            overlay.transient(self.root)
            try:
                overlay.grab_set()
            except Exception:
                pass

            title = tk.Label(
                overlay,
                text=settings.get("title", "LocalPilot Is Using Your PC"),
                font=("Segoe UI", 18, "bold"),
                fg="#f4f6f8",
                bg="#101820",
            )
            title.pack(pady=(24, 12))

            body = tk.Label(
                overlay,
                text=settings.get(
                    "message",
                    "Please do not touch your mouse or keyboard until this action is finished.",
                ),
                font=("Segoe UI", 12),
                fg="#f4f6f8",
                bg="#101820",
                wraplength=500,
                justify="center",
            )
            body.pack(padx=24)

            action = tk.Label(
                overlay,
                text=f"Current action: {action_name}",
                font=("Segoe UI", 11, "bold"),
                fg="#86d0ff",
                bg="#101820",
            )
            action.pack(pady=(14, 8))

            footer = tk.Label(
                overlay,
                text=settings.get(
                    "footer",
                    "LocalPilot will remove this notice as soon as it is safe again.",
                ),
                font=("Segoe UI", 10),
                fg="#b8c4cc",
                bg="#101820",
                wraplength=500,
                justify="center",
            )
            footer.pack(padx=24, pady=(0, 18))

            self.desktop_overlay = overlay

        self.root.after(0, build_overlay)

    def hide_desktop_busy_overlay(self) -> None:
        def destroy_overlay() -> None:
            if self.desktop_overlay is not None and self.desktop_overlay.winfo_exists():
                try:
                    self.desktop_overlay.grab_release()
                except Exception:
                    pass
                self.desktop_overlay.destroy()
            self.desktop_overlay = None

        self.root.after(0, destroy_overlay)


def format_result(result: dict[str, Any]) -> str:
    if "message" in result:
        return str(result["message"])
    if "content" in result:
        return str(result["content"])
    if result.get("ok") and "x" in result and "y" in result:
        return f"Mouse position: ({result['x']}, {result['y']})"
    if result.get("ok") and "path" in result and len(result) <= 3:
        return str(result["path"])
    if "matches" in result:
        matches = result.get("matches") or []
        if not matches:
            return "No matching notes found."
        return "\n".join(f"- {match}" for match in matches)
    if result.get("results"):
        lines = [f"Research results for: {result.get('query', '')}"]
        for item in result["results"]:
            lines.append(f"- {item.get('title', '')}")
            lines.append(f"  {item.get('url', '')}")
            if item.get("snippet"):
                lines.append(f"  {item['snippet']}")
        return "\n".join(lines)
    if "error" in result:
        return f"Error: {result['error']}"
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
