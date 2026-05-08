from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from app.tools import files as file_tools
from app.tools import shell as shell_tools


class CodeMode:
    def __init__(self, app) -> None:
        self.app = app

    def handle(self, request: dict) -> dict:
        text = request["user_text"].strip()
        lowered = text.lower()
        self.app.logger.event("Mode:code", f"Handling code request: {text}")

        if self._is_calculator_scaffold_request(lowered):
            return self._scaffold_calculator_project(text)

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

    def _is_calculator_scaffold_request(self, lowered: str) -> bool:
        return (
            "calculator" in lowered
            and any(word in lowered for word in ("create", "build", "make"))
            and any(hint in lowered for hint in ("folder", " in ", "c:\\", "gui", "double click", "double-click"))
        )

    def _scaffold_calculator_project(self, text: str) -> dict:
        target_dir = self._extract_target_directory(text) or self._default_calculator_project_dir()

        files_to_write = self._calculator_files(Path(target_dir))
        existing_targets = [path for path in files_to_write if Path(path).exists()]
        if existing_targets:
            approved = self.app.ask_approval(
                "Calculator project files already exist and will be overwritten:\n"
                + "\n".join(existing_targets)
            )
            if not approved:
                return {"ok": False, "error": "Calculator project creation cancelled by user."}

        folder_result = file_tools.make_folder(target_dir)
        write_results = []
        for path, content in files_to_write.items():
            write_results.append(file_tools.write_file(path, content))

        return {
            "ok": folder_result.get("ok", False) and all(item.get("ok", False) for item in write_results),
            "message": (
                f"Calculator project created in {target_dir}\n"
                f"Double-click Run Calculator.bat in that folder to start it."
            ),
            "project_path": target_dir,
            "files": list(files_to_write.keys()),
            "write_results": write_results,
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

    def _default_calculator_project_dir(self) -> str:
        base_dir = self.app.root_dir / "workspace" / "generated_apps"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = base_dir / f"CalculatorApp_{timestamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = base_dir / f"CalculatorApp_{timestamp}_{suffix}"
        return str(candidate)

    def _calculator_files(self, target_dir: Path) -> dict[str, str]:
        calculator_py = """import tkinter as tk


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
        display_frame = tk.Frame(self.root, bg="#10131a")
        display_frame.pack(fill="x", padx=16, pady=(16, 8))

        display = tk.Entry(
            display_frame,
            textvariable=self.display_var,
            font=("Segoe UI", 28, "bold"),
            justify="right",
            bd=0,
            relief="flat",
            bg="#1b2230",
            fg="#f5f7fb",
            insertwidth=0,
        )
        display.pack(fill="x", ipady=20)
        display.configure(state="readonly", readonlybackground="#1b2230")

        buttons_frame = tk.Frame(self.root, bg="#10131a")
        buttons_frame.pack(fill="both", expand=True, padx=16, pady=(8, 16))

        layout = [
            ["C", "(", ")", "/"],
            ["7", "8", "9", "*"],
            ["4", "5", "6", "-"],
            ["1", "2", "3", "+"],
            ["0", ".", "=", ""],
        ]

        for row_index, row in enumerate(layout):
            buttons_frame.grid_rowconfigure(row_index, weight=1)
            for col_index, label in enumerate(row):
                buttons_frame.grid_columnconfigure(col_index, weight=1)
                if not label:
                    continue
                self._make_button(buttons_frame, label, row_index, col_index)

    def _make_button(self, parent: tk.Frame, label: str, row: int, column: int) -> None:
        is_operator = label in {"/", "*", "-", "+", "=", "C"}
        bg = "#2a3445" if not is_operator else "#355c7d"
        active_bg = "#42546d" if not is_operator else "#46739a"

        button = tk.Button(
            parent,
            text=label,
            font=("Segoe UI", 20, "bold"),
            bd=0,
            relief="flat",
            bg=bg,
            fg="#f5f7fb",
            activebackground=active_bg,
            activeforeground="#ffffff",
            command=lambda value=label: self._on_press(value),
        )
        button.grid(row=row, column=column, sticky="nsew", padx=6, pady=6, ipady=16)

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
"""

        launcher_bat = """@echo off
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

echo Starting Calculator...
"%PYTHON_EXE%" calculator.py
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Calculator exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
"""

        readme_md = """# CalculatorApp

Simple desktop calculator built with Python and Tkinter.

## Run

Double-click `Run Calculator.bat`.

Or run manually:

```powershell
cd <project folder>
python calculator.py
```

## Files

- `calculator.py`: calculator GUI
- `Run Calculator.bat`: double-click launcher

## Notes

- Uses only Python standard library Tkinter
- No extra package install is required
- If you want a standalone `.exe`, package it later with PyInstaller
"""

        return {
            str(target_dir / "calculator.py"): calculator_py,
            str(target_dir / "Run Calculator.bat"): launcher_bat,
            str(target_dir / "README.md"): readme_md,
        }

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
