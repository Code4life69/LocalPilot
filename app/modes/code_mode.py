from __future__ import annotations

import py_compile
import re
from datetime import datetime
from html import escape
from pathlib import Path

from app.tools import files as file_tools
from app.tools import shell as shell_tools


APP_TEMPLATES = {
    "website": {
        "display_name": "Website",
        "slug": "Website",
        "launcher_name": "Run Website.bat",
        "main_filename": "index.html",
        "readme_name": "README.txt",
    },
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

        if self._looks_like_natural_file_create_request(lowered):
            return self._handle_natural_file_create(text)

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

        response = self.app.ollama.chat_with_role("coder", self.app.system_prompt, text)
        return {"ok": True, "message": response}

    def _looks_like_natural_file_create_request(self, lowered: str) -> bool:
        return (
            any(phrase in lowered for phrase in ("create", "make", "write"))
            and "file" in lowered
            and any(phrase in lowered for phrase in ("named", "called"))
            and any(phrase in lowered for phrase in ("that says", "with", "containing"))
        )

    def _handle_natural_file_create(self, text: str) -> dict:
        parsed = self._parse_natural_file_create_request(text)
        if not parsed:
            return {"ok": False, "error": "Could not determine the file name and content for the file creation request."}
        if parsed.get("error"):
            return {"ok": False, "error": parsed["error"]}

        target_path = parsed["path"]
        content = parsed["content"]
        if self.app.safety.requires_write_confirmation(target_path):
            approved = self.app.ask_approval(f"Overwrite existing file?\n{target_path}")
            if not approved:
                return {"ok": False, "error": "Write cancelled by user."}

        write_result = file_tools.write_file(str(target_path), content)
        if not write_result.get("ok"):
            return write_result

        verification = file_tools.read_file(str(target_path))
        if not verification.get("ok"):
            return {"ok": False, "error": f"File write completed but verification failed for {target_path}."}
        if verification.get("content") != content:
            return {"ok": False, "error": f"File verification mismatch for {target_path}."}

        return {
            "ok": True,
            "message": f"Created file: {target_path}",
            "path": str(target_path),
            "content": content,
        }

    def _is_app_scaffold_request(self, lowered: str) -> bool:
        if not any(word in lowered for word in ("create", "build", "make")):
            return False
        return any(
            hint in lowered
            for hint in (
                "app",
                "program",
                "gui",
                "double click",
                "double-click",
                "starter",
                "website",
                "web page",
                "webpage",
                "landing page",
                "html css",
                "folder",
            )
        )

    def _is_app_verification_request(self, lowered: str) -> bool:
        return lowered.startswith("verify") and "app" in lowered and "run" in lowered

    def _detect_supported_app_kind(self, lowered: str) -> str | None:
        if any(term in lowered for term in ("website", "web page", "webpage", "landing page", "html css", "javascript")):
            return "website"
        if "todo" in lowered:
            return "todo"
        for kind in APP_TEMPLATES:
            if kind in lowered:
                return kind
        return None

    def _scaffold_gui_app(self, text: str, app_kind: str) -> dict:
        target_dir = self._extract_target_directory(text) or self._default_generated_app_dir(app_kind)
        target_path = Path(target_dir)
        website_spec = self._generate_website_spec(text) if app_kind == "website" else None
        files_to_write = self._build_app_files(app_kind, target_path, website_spec)
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
        website_summary = ""
        if website_spec is not None:
            website_summary = (
                f"\nWebsite type: {website_spec['site_type_label']}"
                f"\nTheme: {website_spec['theme_label']}"
                f"\nOpen it by double-clicking {APP_TEMPLATES[app_kind]['launcher_name']}."
            )
        return {
            "ok": folder_result.get("ok", False) and all(item.get("ok", False) for item in write_results),
            "message": (
                f"{display_name} app created in {target_dir}\n"
                f"Double-click {APP_TEMPLATES[app_kind]['launcher_name']} to start it."
                f"{website_summary}"
            ),
            "project_path": target_dir,
            "files": list(files_to_write.keys()),
            "verification": verification,
            "write_results": write_results,
            "site_type": website_spec["site_type"] if website_spec else None,
            "theme": website_spec["theme"] if website_spec else None,
            "generation_mode": website_spec["generation_mode"] if website_spec else None,
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

        extra_files: list[Path] = []
        if app_kind == "website":
            extra_files = [project_path / "style.css", project_path / "script.js"]

        missing = [str(path) for path in (project_path, main_path, launcher_path, readme_path, *extra_files) if not path.exists()]
        if missing:
            return {"ok": False, "error": "Generated app verification failed. Missing files:\n" + "\n".join(missing)}

        syntax_verified = app_kind != "website"
        if app_kind != "website":
            try:
                py_compile.compile(str(main_path), doraise=True)
            except py_compile.PyCompileError as exc:
                return {"ok": False, "error": f"Syntax verification failed for {main_path}: {exc.msg}"}
        else:
            index_content = main_path.read_text(encoding="utf-8")
            missing_links = []
            if 'href="style.css"' not in index_content:
                missing_links.append("style.css link missing from index.html")
            if 'src="script.js"' not in index_content:
                missing_links.append("script.js link missing from index.html")
            if missing_links:
                return {"ok": False, "error": "Generated website verification failed.\n" + "\n".join(missing_links)}

        return {
            "ok": True,
            "project_path": str(project_path),
            "main_file": str(main_path),
            "launcher_file": str(launcher_path),
            "readme_file": str(readme_path),
            "syntax_verified": syntax_verified,
            "static_files_verified": app_kind == "website",
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

    def _build_app_files(self, app_kind: str, target_dir: Path, website_spec: dict | None = None) -> dict[str, str]:
        template = APP_TEMPLATES[app_kind]
        display_name = template["display_name"]
        main_filename = template["main_filename"]
        launcher_name = template["launcher_name"]
        readme_name = template["readme_name"]

        return {
            str(target_dir / main_filename): self._app_main_source(app_kind, display_name, website_spec),
            str(target_dir / launcher_name): self._launcher_source(display_name, main_filename),
            str(target_dir / readme_name): self._readme_source(display_name, launcher_name, main_filename, website_spec),
            **self._extra_app_files(app_kind, target_dir, website_spec),
        }

    def _app_main_source(self, app_kind: str, display_name: str, website_spec: dict | None = None) -> str:
        sources = {
            "website": self._website_html_source(website_spec or self._default_website_spec()),
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
        if display_name == "Website":
            return """@echo off
setlocal
cd /d "%~dp0"

if not exist "index.html" (
    echo index.html was not found.
    pause
    exit /b 1
)

start "" "index.html"
exit /b 0
"""
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

    def _readme_source(
        self,
        display_name: str,
        launcher_name: str,
        main_filename: str,
        website_spec: dict | None = None,
    ) -> str:
        if display_name == "Website":
            website_spec = website_spec or self._default_website_spec()
            return f"""{display_name}
====================

This website scaffold was generated by LocalPilot.

Website type
- {website_spec['site_type_label']}

Theme
- {website_spec['theme_label']}

Prompt focus
- {website_spec['summary']}

Files
- index.html: main page
- style.css: styling
- script.js: client-side behavior
- {launcher_name}: double-click launcher

How to run
1. Open this folder.
2. Double-click {launcher_name}.

Verification
- LocalPilot verified that index.html, style.css, script.js, and {launcher_name} exist.
- LocalPilot verified that index.html links style.css and script.js.
"""
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

    def _extra_app_files(self, app_kind: str, target_dir: Path, website_spec: dict | None = None) -> dict[str, str]:
        if app_kind != "website":
            return {}
        website_spec = website_spec or self._default_website_spec()
        return {
            str(target_dir / "style.css"): self._website_css_source(website_spec),
            str(target_dir / "script.js"): self._website_js_source(website_spec),
        }

    def _generate_website_spec(self, prompt: str) -> dict:
        deterministic_spec = self._build_website_spec(prompt)
        deterministic_spec["generation_mode"] = "deterministic"
        return deterministic_spec

    def _default_website_spec(self) -> dict:
        return self._build_website_spec("make me a basic local website")

    def _build_website_spec(self, prompt: str) -> dict:
        lowered = prompt.lower()
        site_type = self._detect_website_type(lowered)
        theme = self._detect_website_theme(lowered)
        tone = self._detect_website_tone(lowered, site_type)
        context = self._detect_website_context(lowered)
        title_subject = self._build_website_title_subject(lowered, site_type, context)
        palette = self._website_palette(theme, site_type, context)
        layout = self._website_layout(site_type, theme)
        sections = self._website_sections(site_type, context, lowered)
        cta_text = self._website_cta_text(site_type, context)
        footer = self._website_footer(site_type, context)
        hero_heading, subtitle = self._website_hero_copy(site_type, context, theme, title_subject)

        return {
            "site_type": site_type,
            "site_type_label": self._site_type_label(site_type),
            "theme": theme,
            "theme_label": self._theme_label(theme),
            "tone": tone,
            "title": title_subject,
            "hero_heading": hero_heading,
            "subtitle": subtitle,
            "cta_text": cta_text,
            "footer": footer,
            "eyebrow": f"{self._site_type_label(site_type)} | {tone}",
            "sections": sections,
            "palette": palette,
            "layout": layout,
            "summary": self._website_summary(site_type, context, theme),
        }

    def _detect_website_type(self, lowered: str) -> str:
        if "portfolio" in lowered or "projects" in lowered:
            return "portfolio"
        if "lawn care" in lowered or "business" in lowered or "contact section" in lowered or "local website" in lowered:
            return "local_business"
        if "product" in lowered or "landing page" in lowered or "localpilot" in lowered:
            return "product"
        if "ai assistant" in lowered:
            return "landing"
        return "basic"

    def _detect_website_theme(self, lowered: str) -> str:
        if "dark" in lowered or "futuristic" in lowered or "ai assistant" in lowered:
            return "dark"
        return "light"

    def _detect_website_tone(self, lowered: str, site_type: str) -> str:
        if "futuristic" in lowered:
            return "Futuristic"
        if "simple" in lowered or "basic" in lowered:
            return "Simple"
        if site_type == "portfolio":
            return "Confident"
        if site_type == "product":
            return "Focused"
        if site_type == "local_business":
            return "Friendly"
        return "Clean"

    def _detect_website_context(self, lowered: str) -> str:
        if "lawn care" in lowered:
            return "lawn_care"
        if "portfolio" in lowered or "coding projects" in lowered:
            return "coding_portfolio"
        if "localpilot" in lowered:
            return "localpilot"
        if "ai assistant" in lowered:
            return "ai_assistant"
        if "business" in lowered:
            return "business"
        return "generic"

    def _build_website_title_subject(self, lowered: str, site_type: str, context: str) -> str:
        if context == "lawn_care":
            return "FreshCut Lawn Care"
        if context == "coding_portfolio":
            return "Project Portfolio"
        if context == "localpilot":
            return "LocalPilot"
        if context == "ai_assistant":
            return "NeonPilot AI"
        if context == "business":
            return "Summit Business Studio"
        if site_type == "portfolio":
            return "Creative Portfolio"
        if site_type == "product":
            return "Product Launch Site"
        return "Local Website Starter"

    def _website_palette(self, theme: str, site_type: str, context: str) -> dict:
        if context == "lawn_care":
            return {
                "page_bg": "#f4fbf4",
                "surface": "#ffffff",
                "surface_alt": "#e7f6e7",
                "text": "#16311d",
                "muted": "#587160",
                "accent": "#2f9e44",
                "accent_2": "#dff5cf",
                "border": "#cfe7cf",
            }
        if theme == "dark" and context == "ai_assistant":
            return {
                "page_bg": "#08111f",
                "surface": "#0f1b33",
                "surface_alt": "#132544",
                "text": "#eaf3ff",
                "muted": "#9db2cf",
                "accent": "#61dafb",
                "accent_2": "#8b5cf6",
                "border": "#23385e",
            }
        if site_type == "portfolio":
            return {
                "page_bg": "#f6f8fb",
                "surface": "#ffffff",
                "surface_alt": "#eef3ff",
                "text": "#1a2130",
                "muted": "#586278",
                "accent": "#2563eb",
                "accent_2": "#c7d8ff",
                "border": "#d9e2f0",
            }
        if site_type == "product":
            return {
                "page_bg": "#fff8f1",
                "surface": "#ffffff",
                "surface_alt": "#fff1df",
                "text": "#241815",
                "muted": "#6d5a54",
                "accent": "#ef6c00",
                "accent_2": "#ffd9b0",
                "border": "#f2d4bc",
            }
        return {
            "page_bg": "#f7f8fb",
            "surface": "#ffffff",
            "surface_alt": "#eef2f8",
            "text": "#192231",
            "muted": "#5d6b80",
            "accent": "#0f766e",
            "accent_2": "#d4f3ef",
            "border": "#d7e3ea",
        }

    def _website_layout(self, site_type: str, theme: str) -> str:
        if site_type == "portfolio":
            return "project-grid"
        if site_type == "local_business":
            return "service-stack"
        if theme == "dark":
            return "neon-panels"
        if site_type == "product":
            return "feature-spotlight"
        return "clean-sections"

    def _website_sections(self, site_type: str, context: str, lowered: str) -> list[dict[str, str]]:
        if context == "lawn_care":
            return [
                {"title": "Services", "body": "Weekly mowing, edging, seasonal cleanups, and dependable neighborhood scheduling."},
                {"title": "Why Homeowners Call", "body": "Fast estimates, clean finishes, and clear arrival windows that make local service feel easy."},
                {"title": "Contact", "body": "Invite visitors to call, text, or request a same-day quote from the contact section."},
            ]
        if site_type == "portfolio":
            return [
                {"title": "Featured Projects", "body": "Highlight your strongest coding builds with room for screenshots, summaries, and links."},
                {"title": "Skills", "body": "Show the languages, frameworks, and tools you use to ship practical software."},
                {"title": "About", "body": "Use this section to explain how you approach problem-solving, UI, and reliability."},
            ]
        if context == "localpilot":
            return [
                {"title": "Why LocalPilot", "body": "Explain the value of a local Windows assistant that uses guarded tools and local models."},
                {"title": "Core Modes", "body": "Describe chat, coding, research, desktop, and memory as separate trustworthy modes."},
                {"title": "Safety Rules", "body": "Show approvals, visibility, and control boundaries so users understand how actions stay guarded."},
            ]
        if context == "ai_assistant":
            return [
                {"title": "Capabilities", "body": "Introduce desktop awareness, coding help, research, and live tool orchestration."},
                {"title": "Workflow", "body": "Show a clear UI Automation first path with screenshot reasoning only when needed."},
                {"title": "Human Control", "body": "Keep the assistant bold in presentation but explicit about approvals and safety gates."},
            ]
        if "contact" in lowered or site_type == "local_business":
            return [
                {"title": "Services", "body": "Summarize your offer in concrete terms so visitors quickly understand what you do."},
                {"title": "About The Business", "body": "Build trust with a short origin story, values, or proof of reliability."},
                {"title": "Contact", "body": "Include a strong call to reach out, request a quote, or book a conversation."},
            ]
        if site_type == "product":
            return [
                {"title": "Feature Highlights", "body": "Break the product into benefits, workflow advantages, and clear next actions."},
                {"title": "How It Works", "body": "Use a short section to explain the product in plain steps instead of vague hype."},
                {"title": "Get Started", "body": "End with one direct call to action that makes the next click obvious."},
            ]
        return [
            {"title": "Overview", "body": "A clean starter section for describing the website in your own words."},
            {"title": "Highlights", "body": "Use this area for the strongest reasons a visitor should keep reading."},
            {"title": "Next Step", "body": "Close with a direct action, contact option, or launch message."},
        ]

    def _website_cta_text(self, site_type: str, context: str) -> str:
        if context == "lawn_care":
            return "Request A Quote"
        if site_type == "portfolio":
            return "View My Projects"
        if context == "localpilot":
            return "See The Product"
        if context == "ai_assistant":
            return "Enter The Workflow"
        if site_type == "local_business":
            return "Contact The Business"
        if site_type == "product":
            return "Start With The Product"
        return "Explore The Site"

    def _website_footer(self, site_type: str, context: str) -> str:
        if context == "lawn_care":
            return "FreshCut Lawn Care keeps local service simple, clear, and dependable."
        if site_type == "portfolio":
            return "Built to showcase practical coding work and the thinking behind it."
        if context == "localpilot":
            return "LocalPilot focuses on visible, guarded local AI workflows."
        if context == "ai_assistant":
            return "A futuristic presentation with clear human control still built in."
        return "Generated locally by LocalPilot so you can keep building from a working starter."

    def _website_hero_copy(self, site_type: str, context: str, theme: str, title_subject: str) -> tuple[str, str]:
        if context == "lawn_care":
            return (
                "Make your curb appeal feel maintained every week.",
                "This local business layout is tuned for a lawn care service with quick trust cues, service sections, and a contact-focused call to action.",
            )
        if site_type == "portfolio":
            return (
                "Show the projects that prove how you build.",
                "This portfolio layout gives your coding work a clear intro, a featured project area, and a clean place to explain your strengths.",
            )
        if context == "localpilot":
            return (
                "A local product page built for LocalPilot.",
                "This prompt-aware product landing page emphasizes trust, separate modes, and a grounded local workflow instead of vague agent hype.",
            )
        if context == "ai_assistant":
            return (
                "A dark, futuristic shell for an AI assistant.",
                "This version leans into a darker palette, sharper contrast, and a more cinematic presentation while keeping the page simple and static.",
            )
        if site_type == "local_business":
            return (
                f"{title_subject} helps visitors understand your service fast.",
                "This business-style starter keeps the layout practical with service sections, a trust-building middle area, and a visible contact ending.",
            )
        if site_type == "product":
            return (
                f"{title_subject} gets a focused landing page.",
                "This product-oriented starter uses a feature-first layout with an obvious next step and clean sections you can edit immediately.",
            )
        return (
            "A clean local website starter you can edit right away.",
            "This generic version stays simple on purpose while still giving you a hero, supporting sections, and a ready launcher.",
        )

    def _website_summary(self, site_type: str, context: str, theme: str) -> str:
        if context == "lawn_care":
            return "Local business site for a lawn care service."
        if site_type == "portfolio":
            return "Portfolio website for coding projects."
        if context == "localpilot":
            return "Product landing page for LocalPilot."
        if context == "ai_assistant":
            return f"{theme.title()} futuristic website for an AI assistant."
        if site_type == "local_business":
            return "Simple business website with contact-focused sections."
        if site_type == "product":
            return "Product page with clear feature and CTA sections."
        return "Generic basic website starter."

    def _site_type_label(self, site_type: str) -> str:
        labels = {
            "landing": "Landing Page",
            "portfolio": "Portfolio",
            "local_business": "Local Business Site",
            "product": "Product Page",
            "basic": "Generic Basic Site",
        }
        return labels[site_type]

    def _theme_label(self, theme: str) -> str:
        return "Dark" if theme == "dark" else "Light"

    def _website_html_source(self, website_spec: dict) -> str:
        sections_html = "\n".join(
            f"""        <section class="content-card">
            <h2>{escape(section['title'])}</h2>
            <p>{escape(section['body'])}</p>
        </section>"""
            for section in website_spec["sections"]
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(website_spec['title'])}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body data-site-type="{website_spec['site_type']}" data-theme="{website_spec['theme']}" data-layout="{website_spec['layout']}">
    <header class="hero-shell">
        <div class="hero-card">
            <p class="eyebrow">{escape(website_spec['eyebrow'])}</p>
            <h1>{escape(website_spec['hero_heading'])}</h1>
            <p class="lead">{escape(website_spec['subtitle'])}</p>
            <div class="hero-actions">
                <button id="ctaButton" class="primary-button">{escape(website_spec['cta_text'])}</button>
                <span id="statusText" class="status-text">Website ready.</span>
            </div>
        </div>
    </header>
    <main class="content-shell">
{sections_html}
    </main>
    <footer class="site-footer">
        <p>{escape(website_spec['footer'])}</p>
    </footer>
    <script src="script.js"></script>
</body>
</html>
"""

    def _website_css_source(self, website_spec: dict) -> str:
        palette = website_spec["palette"]
        return f"""* {{
    box-sizing: border-box;
}}

:root {{
    --page-bg: {palette['page_bg']};
    --surface: {palette['surface']};
    --surface-alt: {palette['surface_alt']};
    --text: {palette['text']};
    --muted: {palette['muted']};
    --accent: {palette['accent']};
    --accent-2: {palette['accent_2']};
    --border: {palette['border']};
}}

body {{
    margin: 0;
    min-height: 100vh;
    font-family: "Segoe UI", sans-serif;
    background: radial-gradient(circle at top, var(--accent-2), var(--page-bg) 42%);
    color: var(--text);
}}

.hero-shell {{
    padding: 48px 24px 20px;
}}

.hero-card,
.content-card {{
    width: min(1080px, 100%);
    margin: 0 auto;
    border: 1px solid var(--border);
    border-radius: 24px;
    background: var(--surface);
    box-shadow: 0 20px 55px rgba(15, 23, 42, 0.12);
}}

.hero-card {{
    padding: 40px;
}}

.content-shell {{
    display: grid;
    gap: 18px;
    padding: 0 24px 32px;
}}

.content-card {{
    padding: 28px;
    background: var(--surface-alt);
}}

.eyebrow {{
    margin: 0 0 12px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 700;
}}

h1 {{
    margin: 0 0 14px;
    font-size: clamp(2.3rem, 5vw, 4.4rem);
    line-height: 1.05;
}}

h2 {{
    margin: 0 0 10px;
    font-size: 1.45rem;
}}

.lead,
.content-card p,
.status-text,
.site-footer p {{
    line-height: 1.7;
    font-size: 1rem;
}}

.lead,
.content-card p,
.status-text {{
    color: var(--muted);
}}

.hero-actions {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
    margin-top: 22px;
}}

.primary-button {{
    border: none;
    border-radius: 999px;
    padding: 14px 22px;
    background: var(--accent);
    color: {"#07111f" if website_spec['theme'] == "dark" else "#ffffff"};
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
}}

.site-footer {{
    padding: 0 24px 36px;
}}

.site-footer p {{
    width: min(1080px, 100%);
    margin: 0 auto;
    color: var(--muted);
}}

@media (min-width: 860px) {{
    .content-shell {{
        grid-template-columns: repeat({3 if website_spec['site_type'] in {'portfolio', 'product'} else 1}, minmax(0, 1fr));
    }}

    .content-card:last-child {{
        {"grid-column: span 3;" if website_spec['site_type'] in {'portfolio', 'product'} else ""}
    }}
}}
"""

    def _website_js_source(self, website_spec: dict) -> str:
        return f"""const button = document.getElementById("ctaButton");
const statusText = document.getElementById("statusText");

button.addEventListener("click", () => {{
    statusText.textContent = "{website_spec['cta_text']} clicked. This {website_spec['site_type_label'].lower()} is ready for your edits.";
}});
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

    def _parse_natural_file_create_request(self, text: str) -> dict | None:
        lowered = text.lower()
        explicit_file_path = self._extract_explicit_file_path(text)
        filename = None
        if explicit_file_path:
            target_path = explicit_file_path
        else:
            filename = self._extract_named_filename(text)
            if not filename:
                return None
            base_dir = self._workspace_root()
            target_path = base_dir / filename
            if not target_path.resolve().is_relative_to(base_dir.resolve()):
                return {"error": f"Refusing to write outside workspace without an explicit path: {target_path}"}

        content = self._extract_file_content(text)
        if content is None:
            return None

        if isinstance(target_path, dict):
            return target_path
        return {"path": target_path, "content": content}

    def _extract_named_filename(self, text: str) -> str | None:
        patterns = [
            r'\b(?:named|called)\s+"?([^"\n]+?\.[A-Za-z0-9]+)"?(?:\s|$)',
            r'\bfile\s+named\s+"?([^"\n]+?\.[A-Za-z0-9]+)"?(?:\s|$)',
            r'\bfile\s+called\s+"?([^"\n]+?\.[A-Za-z0-9]+)"?(?:\s|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,")
        return None

    def _extract_explicit_file_path(self, text: str) -> Path | None:
        quoted = re.findall(r'"([^"]+)"', text)
        for candidate in quoted:
            if re.match(r"^[A-Za-z]:\\.+\.[A-Za-z0-9]+$", candidate):
                return Path(candidate)
        match = re.search(r"([A-Za-z]:\\[A-Za-z0-9_ .\\-]+\.[A-Za-z0-9]+)", text)
        if match:
            return Path(match.group(1).strip().rstrip(". "))
        return None

    def _extract_file_content(self, text: str) -> str | None:
        patterns = [
            r"\bthat says\s+(.+)$",
            r"\bwith\s+(.+)$",
            r"\bcontaining\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                return value.strip('"')
        return None

    def _workspace_root(self) -> Path:
        return self.app.root_dir / "workspace"
