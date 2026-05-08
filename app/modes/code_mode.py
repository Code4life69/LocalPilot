from __future__ import annotations

import py_compile
import re
from datetime import datetime
from pathlib import Path

from app.tools import files as file_tools
from app.tools import shell as shell_tools


APP_TEMPLATES = {
    "calculator": {
        "display_name": "Calculator",
        "slug": "CalculatorApp",
        "launcher_name": "Run Calculator.bat",
        "main_filename": "main.py",
        "readme_name": "README.txt",
    },
    "notepad": {
        "display_name": "Notepad",
        "slug": "NotepadApp",
        "launcher_name": "Run Notepad.bat",
        "main_filename": "main.py",
        "readme_name": "README.txt",
    },
    "todo": {
        "display_name": "Todo List",
        "slug": "TodoListApp",
        "launcher_name": "Run Todo List.bat",
        "main_filename": "main.py",
        "readme_name": "README.txt",
    },
    "timer": {
        "display_name": "Timer",
        "slug": "TimerApp",
        "launcher_name": "Run Timer.bat",
        "main_filename": "main.py",
        "readme_name": "README.txt",
    },
}


class CodeMode:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        lowered = text.lower()
        self.app.logger.event("Mode:code", f"Handling code request: {text}")

        if self._is_app_verification_request(lowered):
            return self._verify_generated_app(text)

        app_kind = self._detect_supported_app_kind(lowered)
        if app_kind and self._is_app_scaffold_request(lowered):
            return self._scaffold_gui_app(text, app_kind)

        if lowered.startswith("list ") or "list folder" in lowered or "list files" in lowered:
            path = self._extract_path(text) or "."
            return file_tools.list_folder(path)

        if lowered.startswith("read ") or "read file" in lowered:
            path = self._extract_path(text)
            if not path:
                return {"ok": False, "error": "No file path provided."}
            return file_tools.read_file(path)

        if lowered.startswith("write ") or "write file" in lowered:
            path, content = self._extract_write_args(text)
            if not path:
                return {"ok": False, "error": "No file path provided for write."}
            if self.app.safety.requires_write_confirmation(path):
                approved = self.app.ask_approval(f"Overwrite existing file?\n{path}")
                if not approved:
                    return {"ok": False, "error": "Write cancelled by user."}
            return file_tools.write_file(path, content)

        if lowered.startswith("append ") or "append file" in lowered:
            path, content = self._extract_write_args(text)
            if not path:
                return {"ok": False, "error": "No file path provided for append."}
            return file_tools.append_file(path, content)

        if lowered.startswith("mkdir ") or "make folder" in lowered or "create folder" in lowered:
            path = self._extract_path(text)
            if not path:
                return {"ok": False, "error": "No folder path provided."}
            return file_tools.make_folder(path)

        if lowered.startswith("copy ") or "copy file" in lowered:
            src, dst = self._extract_two_paths(text)
            if not src or not dst:
                return {"ok": False, "error": "Copy requires source and destination."}
            if self.app.safety.requires_move_confirmation(dst):
                approved = self.app.ask_approval(f"Destination exists. Copy and overwrite?\n{dst}")
                if not approved:
                    return {"ok": False, "error": "Copy cancelled by user."}
            return file_tools.copy_file(src, dst)

        if lowered.startswith("move ") or "move file" in lowered:
            src, dst = self._extract_two_paths(text)
            if not src or not dst:
                return {"ok": False, "error": "Move requires source and destination."}
            approved = self.app.ask_approval(f"Approve file move?\n{src}\n->\n{dst}")
            if not approved:
                return {"ok": False, "error": "Move cancelled by user."}
            return file_tools.move_file(src, dst)

        if lowered.startswith("run ") or lowered.startswith("shell ") or "run command" in lowered:
            command = self._extract_command(text)
            if not command:
                return {"ok": False, "error": "No command provided."}
            if self.app.safety.is_command_blocked(command):
                return {"ok": False, "error": f"Blocked dangerous command: {command}"}
            approved = self.app.ask_approval(f"Command wants to run:\n{command}")
            if not approved:
                return {"ok": False, "error": "Command cancelled by user."}
            return shell_tools.run_command(command, cwd=str(Path.cwd()))

        response = self.app.ollama.chat(self.app.system_prompt, text)
        return {"ok": True, "message": response}

    def _is_app_scaffold_request(self, lowered: str) -> bool:
        return any(word in lowered for word in ("create", "build", "make")) and any(
            hint in lowered for hint in ("app", "program", "gui", "double click", "double-click", "starter")
        )

    def _is_app_verification_request(self, lowered: str) -> bool:
        return lowered.startswith("verify") and "app" in lowered and "run" in lowered

    def _detect_supported_app_kind(self, lowered: str) -> str | None:
        if "todo" in lowered:
            return "todo"
        for kind in APP_TEMPLATES:
            if kind in lowered:
                return kind
        return None

    def _scaffold_gui_app(self, text: str, app_kind: str) -> dict:
        target_dir = self._extract_target_directory(text) or self._default_generated_app_dir(app_kind)
        target_path = Path(target_dir)
        files_to_write = self._build_app_files(app_kind, target_path)
        existing_targets = [path for path in files_to_write if Path(path).exists()]
        if existing_targets:
            approved = self.app.ask_approval(
                f"{APP_TEMPLATES[app_kind]['display_name']} app files already exist and will be overwritten:\n"
                + "\n".join(existing_targets)
            )
            if not approved:
                return {"ok": False, "error": "App creation cancelled by user."}

        folder_result = file_tools.make_folder(target_dir)
        write_results = [file_tools.write_file(path, content) for path, content in files_to_write.items()]
        verification = self._verify_app_outputs(target_path, app_kind)
        if not verification["ok"]:
            return verification

        display_name = APP_TEMPLATES[app_kind]["display_name"]
        return {
            "ok": folder_result.get("ok", False) and all(item.get("ok", False) for item in write_results),
            "message": (
                f"{display_name} app created in {target_dir}\n"
                f"Double-click {APP_TEMPLATES[app_kind]['launcher_name']} to start it."
            ),
            "project_path": target_dir,
            "files": list(files_to_write.keys()),
            "verification": verification,
            "write_results": write_results,
        }

    def _verify_generated_app(self, text: str) -> dict:
        app_kind = self._detect_supported_app_kind(text.lower())
        target_dir = self._extract_target_directory(text)
        if target_dir:
            project_path = Path(target_dir)
        else:
            project_path = self._find_latest_generated_app(app_kind)
        if project_path is None:
            return {"ok": False, "error": "Could not find a generated app to verify."}
        if app_kind is None:
            app_kind = self._infer_app_kind_from_path(project_path)
        if app_kind is None:
            return {"ok": False, "error": f"Could not determine app type for {project_path}."}

        verification = self._verify_app_outputs(project_path, app_kind)
        if not verification["ok"]:
            return verification

        return {
            "ok": True,
            "message": (
                f"Verified {APP_TEMPLATES[app_kind]['display_name']} app in {project_path}\n"
                f"Run it by double-clicking {APP_TEMPLATES[app_kind]['launcher_name']}."
            ),
            "project_path": str(project_path),
            "verification": verification,
        }

    def _verify_app_outputs(self, project_path: Path, app_kind: str) -> dict:
        template = APP_TEMPLATES[app_kind]
        main_path = project_path / template["main_filename"]
        launcher_path = project_path / template["launcher_name"]
        readme_path = project_path / template["readme_name"]

        missing = [str(path) for path in (project_path, main_path, launcher_path, readme_path) if not path.exists()]
        if missing:
            return {"ok": False, "error": "Generated app verification failed. Missing files:\n" + "\n".join(missing)}

        try:
            py_compile.compile(str(main_path), doraise=True)
        except py_compile.PyCompileError as exc:
            return {"ok": False, "error": f"Syntax verification failed for {main_path}: {exc.msg}"}

        return {
            "ok": True,
            "project_path": str(project_path),
            "main_file": str(main_path),
            "launcher_file": str(launcher_path),
            "readme_file": str(readme_path),
            "syntax_verified": True,
        }

    def _extract_target_directory(self, text: str) -> str | None:
        quoted = re.findall(r'"([^"]+)"', text)
        for candidate in quoted:
            if ":" in candidate or "\\" in candidate or "/" in candidate:
                return str(Path(candidate))

        path_match = re.search(r"([A-Za-z]:\\[A-Za-z0-9_ .\\-]+)", text)
        if path_match:
            raw_path = path_match.group(1).strip()
            raw_path = re.split(r"(?<=[A-Za-z0-9_])\.\s+[A-Z]", raw_path)[0]
            raw_path = raw_path.rstrip(". ")
            return raw_path
        return None

    def _default_generated_app_dir(self, app_kind: str) -> str:
        base_dir = self.app.root_dir / "workspace" / "generated_apps"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = APP_TEMPLATES[app_kind]["slug"]
        candidate = base_dir / f"{slug}_{timestamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = base_dir / f"{slug}_{timestamp}_{suffix}"
        return str(candidate)

    def _find_latest_generated_app(self, app_kind: str | None) -> Path | None:
        base_dir = self.app.root_dir / "workspace" / "generated_apps"
        if not base_dir.exists():
            return None
        candidates = [path for path in base_dir.iterdir() if path.is_dir()]
        if app_kind is not None:
            prefix = APP_TEMPLATES[app_kind]["slug"]
            candidates = [path for path in candidates if path.name.startswith(prefix)]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _infer_app_kind_from_path(self, project_path: Path) -> str | None:
        lowered = project_path.name.lower()
        for kind, template in APP_TEMPLATES.items():
            if template["slug"].lower() in lowered:
                return kind
        return None

    def _build_app_files(self, app_kind: str, target_dir: Path) -> dict[str, str]:
        template = APP_TEMPLATES[app_kind]
        display_name = template["display_name"]
        main_filename = template["main_filename"]
        launcher_name = template["launcher_name"]
        readme_name = template["readme_name"]

        return {
            str(target_dir / main_filename): self._app_main_source(app_kind, display_name),
            str(target_dir / launcher_name): self._launcher_source(display_name, main_filename),
            str(target_dir / readme_name): self._readme_source(display_name, launcher_name, main_filename),
        }

    def _app_main_source(self, app_kind: str, display_name: str) -> str:
        sources = {
            "calculator": """import tkinter as tk


class CalculatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Calculator")
        self.root.geometry("360x520")
        self.root.resizable(False, False)
        self.expression = ""
        self.display_var = tk.StringVar(value="0")
        self._build_ui()

    def _build_ui(self) -> None:
        self.root.configure(bg="#10131a")
        display = tk.Entry(
            self.root,
            textvariable=self.display_var,
            font=("Segoe UI", 28, "bold"),
            justify="right",
            bd=0,
            relief="flat",
            bg="#1b2230",
            fg="#f5f7fb",
            insertwidth=0,
            readonlybackground="#1b2230",
        )
        display.pack(fill="x", padx=16, pady=(16, 12), ipady=20)
        display.configure(state="readonly")

        grid = tk.Frame(self.root, bg="#10131a")
        grid.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        layout = [
            ["C", "(", ")", "/"],
            ["7", "8", "9", "*"],
            ["4", "5", "6", "-"],
            ["1", "2", "3", "+"],
            ["0", ".", "=", ""],
        ]
        for row_index, row in enumerate(layout):
            grid.grid_rowconfigure(row_index, weight=1)
            for col_index, label in enumerate(row):
                grid.grid_columnconfigure(col_index, weight=1)
                if not label:
                    continue
                button = tk.Button(
                    grid,
                    text=label,
                    font=("Segoe UI", 20, "bold"),
                    bd=0,
                    relief="flat",
                    bg="#355c7d" if label in {"/", "*", "-", "+", "=", "C"} else "#2a3445",
                    fg="#f5f7fb",
                    activebackground="#46739a",
                    activeforeground="#ffffff",
                    command=lambda value=label: self._on_press(value),
                )
                button.grid(row=row_index, column=col_index, sticky="nsew", padx=6, pady=6, ipady=16)

    def _on_press(self, value: str) -> None:
        if value == "C":
            self.expression = ""
            self.display_var.set("0")
            return
        if value == "=":
            self._evaluate()
            return
        if self.display_var.get() == "0" and value not in {".", "+", "-", "*", "/", ")"}:
            self.expression = value
        else:
            self.expression += value
        self.display_var.set(self.expression)

    def _evaluate(self) -> None:
        try:
            result = eval(self.expression, {"__builtins__": {}}, {})
            self.expression = str(result)
            self.display_var.set(self.expression)
        except Exception:
            self.expression = ""
            self.display_var.set("Error")


def main() -> None:
    root = tk.Tk()
    CalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
""",
            "notepad": """import tkinter as tk
from tkinter import filedialog, messagebox


class NotepadApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Notepad")
        self.root.geometry("760x520")
        self.current_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root, bg="#dfe5ec")
        toolbar.pack(fill="x")

        for label, command in (("New", self.new_file), ("Open", self.open_file), ("Save", self.save_file)):
            tk.Button(toolbar, text=label, command=command, width=10).pack(side="left", padx=6, pady=6)

        self.editor = tk.Text(self.root, wrap="word", font=("Segoe UI", 11))
        self.editor.pack(fill="both", expand=True)

    def new_file(self) -> None:
        self.current_path = None
        self.editor.delete("1.0", tk.END)
        self.root.title("Notepad")

    def open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", content)
        self.current_path = path
        self.root.title(f"Notepad - {path}")

    def save_file(self) -> None:
        path = self.current_path
        if path is None:
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.editor.get("1.0", tk.END))
        self.current_path = path
        self.root.title(f"Notepad - {path}")
        messagebox.showinfo("Notepad", f"Saved to {path}")


def main() -> None:
    root = tk.Tk()
    NotepadApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
""",
            "todo": """import tkinter as tk


class TodoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Todo List")
        self.root.geometry("520x460")
        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, padx=12, pady=12)
        top.pack(fill="x")

        self.entry = tk.Entry(top, font=("Segoe UI", 11))
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _event: self.add_item())

        tk.Button(top, text="Add", command=self.add_item, width=10).pack(side="left", padx=(8, 0))

        self.listbox = tk.Listbox(self.root, font=("Segoe UI", 11), selectmode=tk.SINGLE)
        self.listbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        bottom = tk.Frame(self.root, padx=12, pady=(0, 12))
        bottom.pack(fill="x")
        tk.Button(bottom, text="Remove Selected", command=self.remove_selected).pack(side="left")
        tk.Button(bottom, text="Clear All", command=self.clear_all).pack(side="left", padx=(8, 0))

    def add_item(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.listbox.insert(tk.END, text)
        self.entry.delete(0, tk.END)

    def remove_selected(self) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        self.listbox.delete(selection[0])

    def clear_all(self) -> None:
        self.listbox.delete(0, tk.END)


def main() -> None:
    root = tk.Tk()
    TodoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
""",
            "timer": """import tkinter as tk


class TimerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Timer")
        self.root.geometry("360x280")
        self.remaining = 0
        self.running = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.display = tk.StringVar(value="00:00")
        tk.Label(self.root, text="Countdown Timer", font=("Segoe UI", 18, "bold")).pack(pady=(20, 12))
        tk.Label(self.root, textvariable=self.display, font=("Consolas", 34, "bold")).pack(pady=(0, 16))

        entry_row = tk.Frame(self.root)
        entry_row.pack(pady=(0, 14))
        tk.Label(entry_row, text="Seconds:", font=("Segoe UI", 11)).pack(side="left")
        self.seconds_entry = tk.Entry(entry_row, width=8, font=("Segoe UI", 11))
        self.seconds_entry.pack(side="left", padx=(8, 0))

        buttons = tk.Frame(self.root)
        buttons.pack()
        tk.Button(buttons, text="Start", command=self.start_timer, width=10).pack(side="left", padx=4)
        tk.Button(buttons, text="Stop", command=self.stop_timer, width=10).pack(side="left", padx=4)
        tk.Button(buttons, text="Reset", command=self.reset_timer, width=10).pack(side="left", padx=4)

    def start_timer(self) -> None:
        if not self.running:
            if self.remaining <= 0:
                try:
                    self.remaining = max(0, int(self.seconds_entry.get().strip()))
                except ValueError:
                    self.remaining = 0
            if self.remaining > 0:
                self.running = True
                self._tick()

    def stop_timer(self) -> None:
        self.running = False

    def reset_timer(self) -> None:
        self.running = False
        self.remaining = 0
        self.display.set("00:00")

    def _tick(self) -> None:
        mins, secs = divmod(self.remaining, 60)
        self.display.set(f"{mins:02d}:{secs:02d}")
        if not self.running:
            return
        if self.remaining <= 0:
            self.running = False
            self.root.bell()
            return
        self.remaining -= 1
        self.root.after(1000, self._tick)


def main() -> None:
    root = tk.Tk()
    TimerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
""",
        }
        return sources[app_kind]

    def _launcher_source(self, display_name: str, main_filename: str) -> str:
        return f"""@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="

if exist ".venv\\Scripts\\python.exe" (
    set "PYTHON_EXE=.venv\\Scripts\\python.exe"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_EXE=py"
    ) else (
        where python >nul 2>nul
        if %errorlevel%==0 (
            set "PYTHON_EXE=python"
        )
    )
)

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Install Python or create a local virtual environment first.
    pause
    exit /b 1
)

echo Starting {display_name}...
"%PYTHON_EXE%" {main_filename}
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo {display_name} exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
"""

    def _readme_source(self, display_name: str, launcher_name: str, main_filename: str) -> str:
        return f"""{display_name}
====================

This app was generated by LocalPilot.

Files
- {main_filename}: main GUI application
- {launcher_name}: double-click launcher

How to run
1. Open this folder.
2. Double-click {launcher_name}.

Manual run
- python {main_filename}

Verification
- LocalPilot verified that {main_filename} exists.
- LocalPilot ran Python syntax verification on {main_filename}.
"""

    def _extract_path(self, text: str) -> str | None:
        match = re.search(r'"([^"]+)"', text)
        if match:
            return match.group(1)
        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            return parts[1].replace("folder", "").replace("file", "").strip()
        return None

    def _extract_write_args(self, text: str) -> tuple[str | None, str]:
        quoted = re.findall(r'"([^"]+)"', text)
        if len(quoted) >= 2:
            return quoted[0], quoted[1]
        parts = text.split(maxsplit=2)
        if len(parts) >= 3:
            return parts[1], parts[2]
        if len(parts) == 2:
            return parts[1], ""
        return None, ""

    def _extract_two_paths(self, text: str) -> tuple[str | None, str | None]:
        quoted = re.findall(r'"([^"]+)"', text)
        if len(quoted) >= 2:
            return quoted[0], quoted[1]
        parts = text.split(maxsplit=2)
        if len(parts) >= 3:
            return parts[1], parts[2]
        return None, None

    def _extract_command(self, text: str) -> str:
        lowered = text.lower()
        for prefix in ("run command", "run", "shell"):
            if lowered.startswith(prefix):
                return text[len(prefix):].strip()
        return text.strip()
