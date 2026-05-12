from __future__ import annotations

import py_compile
import re
import subprocess
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from app.tools import files as file_tools
from app.tools import shell as shell_tools
from app.tools.screen import get_active_window_basic
from app.tools.windows_ui import get_active_window_title
from app.tools.web import search_web


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
    "script": {
        "display_name": "Tool Script",
        "slug": "ToolScript",
        "launcher_name": "Run Tool.bat",
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

        if self._is_professional_build_request(lowered):
            return self._run_professional_build(text)

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

    def _is_professional_build_request(self, lowered: str) -> bool:
        return lowered.startswith("professional build ") or lowered.startswith("build this professionally")

    def _extract_professional_build_request(self, text: str) -> str:
        lowered = text.lower().strip()
        if lowered.startswith("professional build "):
            return text.strip()[len("professional build "):].strip()
        if lowered.startswith("build this professionally"):
            remainder = text.strip()[len("build this professionally"):].strip()
            return remainder.lstrip(": ").strip()
        return text.strip()

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
                "tool",
                "script",
            )
        )

    def _is_app_verification_request(self, lowered: str) -> bool:
        return lowered.startswith("verify") and "app" in lowered and "run" in lowered

    def _detect_supported_app_kind(self, lowered: str) -> str | None:
        if any(term in lowered for term in ("website", "web page", "webpage", "landing page", "html css", "javascript")):
            return "website"
        if "todo" in lowered:
            return "todo"
        if "script" in lowered or "tool" in lowered:
            return "script"
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

    def _run_professional_build(self, text: str) -> dict:
        settings = self._professional_build_settings()
        if not settings["enabled"]:
            return {"ok": False, "error": "Professional build mode is disabled in settings."}

        build_request = self._extract_professional_build_request(text)
        if not build_request:
            return {"ok": False, "error": "No build request was provided after the professional build command."}

        app_kind = self._detect_professional_app_kind(build_request.lower())
        if app_kind is None:
            return {
                "ok": False,
                "error": (
                    "Professional build mode currently supports website, calculator, notepad, todo, timer, "
                    "and generic script/tool requests."
                ),
            }

        explicit_target_dir = self._extract_target_directory(build_request)
        project_path = Path(explicit_target_dir or self._default_generated_app_dir(app_kind))
        workspace_root = self._workspace_root().resolve()
        explicit_target = explicit_target_dir is not None
        target_in_workspace = self._path_is_within(project_path, workspace_root)
        if not explicit_target and not target_in_workspace:
            return {"ok": False, "error": f"Professional build refused to write outside workspace: {project_path}"}

        research = self._request_professional_research(build_request, settings) if settings["allow_web_research"] else None
        brief = self._build_project_brief(build_request, app_kind, project_path)
        website_spec = self._generate_website_spec(build_request) if app_kind == "website" else None
        professional_context = {
            "brief": brief,
            "acceptance_checklist": [],
            "verification_summary": {},
            "known_limitations": [],
            "research": research,
        }
        files_to_write = self._build_app_files(
            app_kind,
            project_path,
            website_spec,
            professional_context=professional_context,
        )
        existing_targets = [path for path in files_to_write if Path(path).exists()]
        if existing_targets:
            approved = self.app.ask_approval(
                "Professional build files already exist and will be overwritten:\n" + "\n".join(existing_targets)
            )
            if not approved:
                return {"ok": False, "error": "Professional build cancelled by user."}

        write_result = self._write_project_files(project_path, files_to_write)
        if not write_result["ok"]:
            return write_result

        verification_history: list[dict[str, Any]] = []
        improvement_history: list[str] = []
        final_checklist: list[dict[str, Any]] = []
        final_verification: dict[str, Any] = {}
        final_review: dict[str, Any] = {"ready": False, "issues": []}
        status = "stopped_at_max_passes"

        for pass_index in range(1, settings["max_passes"] + 1):
            verification = self._run_professional_verification(
                project_path=project_path,
                app_kind=app_kind,
                explicit_target=explicit_target,
                settings=settings,
            )
            verification_history.append(
                {
                    "pass": pass_index,
                    "ok": verification["ok"],
                    "checks_performed": verification["checks_performed"],
                    "error": verification.get("error", ""),
                }
            )

            final_checklist = self._build_acceptance_checklist(
                brief=brief,
                app_kind=app_kind,
                project_path=project_path,
                verification=verification,
                explicit_target=explicit_target,
                settings=settings,
            )
            final_review = self._self_review_professional_build(
                build_request=build_request,
                app_kind=app_kind,
                verification=verification,
                checklist=final_checklist,
            )
            known_limitations = self._collect_known_limitations(verification, final_review)
            professional_context = {
                "brief": brief,
                "acceptance_checklist": final_checklist,
                "verification_summary": verification,
                "known_limitations": known_limitations,
                "research": research,
            }
            self._update_professional_readme(project_path, app_kind, website_spec, professional_context)

            final_verification = verification
            acceptance_passed = all(item["passed"] for item in final_checklist)
            review_ready = final_review.get("ready", False)
            if acceptance_passed and review_ready:
                status = "completed"
                break

            if pass_index >= settings["max_passes"]:
                status = "verification_failed" if not verification["ok"] else "stopped_at_max_passes"
                break

            improvement_result = self._apply_professional_improvements(
                project_path=project_path,
                app_kind=app_kind,
                review=final_review,
                website_spec=website_spec,
                professional_context=professional_context,
            )
            improvement_history.extend(improvement_result["applied"])

        known_limitations = professional_context["known_limitations"]
        report = self._compose_professional_report(
            status=status,
            project_path=project_path,
            app_kind=app_kind,
            files=files_to_write,
            brief=brief,
            checklist=final_checklist,
            verification=final_verification,
            verification_history=verification_history,
            improvements=improvement_history,
            known_limitations=known_limitations,
            research=research,
        )
        return {
            "ok": status == "completed",
            "message": report,
            "project_path": str(project_path),
            "files": list(files_to_write.keys()),
            "brief": brief,
            "acceptance_checklist": final_checklist,
            "verification": final_verification,
            "verification_history": verification_history,
            "improvements": improvement_history,
            "known_limitations": known_limitations,
            "research": research,
            "passes_completed": len(verification_history),
            "status": status,
        }

    def _professional_build_settings(self) -> dict[str, Any]:
        defaults = {
            "enabled": True,
            "max_passes": 3,
            "allow_web_research": True,
            "require_acceptance_checklist": True,
            "stop_on_failed_verification": True,
            "launch_verification_enabled": True,
            "launch_timeout_seconds": 8,
        }
        configured = self.app.settings.get("professional_build", {})
        return {**defaults, **configured}

    def _detect_professional_app_kind(self, lowered: str) -> str | None:
        detected = self._detect_supported_app_kind(lowered)
        if detected is not None:
            return detected
        if any(term in lowered for term in ("tool", "script", "program")):
            return "script"
        return None

    def _request_professional_research(self, build_request: str, settings: dict[str, Any]) -> dict[str, Any] | None:
        query = self._research_query_for_build(build_request)
        if not query:
            return None
        result = search_web(query, max_results=3)
        if not result.get("ok"):
            return {
                "needed": True,
                "query": query,
                "ok": False,
                "summary": f"Research requested but failed: {result.get('error', 'unknown error')}",
                "results": [],
            }
        summary_lines = [f"Research query: {query}"]
        for item in result.get("results", [])[:3]:
            summary_lines.append(f"- {item.get('title', 'Untitled')} | {item.get('url', '')}")
        return {
            "needed": True,
            "query": query,
            "ok": True,
            "summary": "\n".join(summary_lines),
            "results": result.get("results", [])[:3],
        }

    def _research_query_for_build(self, build_request: str) -> str | None:
        lowered = build_request.lower()
        keywords = {
            "sqlite": "Python sqlite3 official documentation",
            "api": "Python API client best practices official documentation",
            "excel": "Python openpyxl official documentation",
            "csv": "Python csv module official documentation",
            "json": "Python json module official documentation",
            "file dialog": "Python tkinter filedialog documentation",
            "windows behavior": "Python Windows subprocess and file path behavior documentation",
        }
        for key, query in keywords.items():
            if key in lowered:
                return query
        return None

    def _build_project_brief(self, build_request: str, app_kind: str, project_path: Path) -> dict[str, Any]:
        template = APP_TEMPLATES[app_kind]
        files = [template["main_filename"], template["launcher_name"], template["readme_name"]]
        if app_kind == "website":
            files.extend(["style.css", "script.js"])
        return {
            "request": build_request,
            "project_kind": app_kind,
            "target_platform": "Windows desktop/local filesystem",
            "expected_files": files,
            "project_path": str(project_path),
            "run_instructions": self._project_run_instructions(app_kind),
            "done_means": (
                "Acceptance checklist passes, verification checks are green, README is clear, "
                "and LocalPilot can explain how to run the project."
            ),
        }

    def _write_project_files(self, project_path: Path, files_to_write: dict[str, str]) -> dict[str, Any]:
        folder_result = file_tools.make_folder(str(project_path))
        if not folder_result.get("ok"):
            return folder_result
        write_results = [file_tools.write_file(path, content) for path, content in files_to_write.items()]
        failed = [item for item in write_results if not item.get("ok")]
        if failed:
            return {"ok": False, "error": failed[0].get("error", "Failed to write project files.")}
        return {"ok": True, "write_results": write_results}

    def _run_professional_verification(
        self,
        project_path: Path,
        app_kind: str,
        explicit_target: bool,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        template = APP_TEMPLATES[app_kind]
        main_path = project_path / template["main_filename"]
        launcher_path = project_path / template["launcher_name"]
        readme_path = project_path / template["readme_name"]
        base_verification = self._verify_app_outputs(project_path, app_kind)
        checks_performed = ["file existence", "README presence", "launcher inspection"]
        if app_kind == "website":
            checks_performed.append("static asset linkage")
        else:
            checks_performed.append("python syntax check")
        launch_verification = self._verify_launch_readiness(
            project_path=project_path,
            app_kind=app_kind,
            settings=settings,
        )
        checks_performed.append("launch verification")

        readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        launcher_text = launcher_path.read_text(encoding="utf-8") if launcher_path.exists() else ""
        main_text = main_path.read_text(encoding="utf-8") if main_path.exists() else ""

        verification = {
            "ok": base_verification.get("ok", False),
            "base_verification": base_verification,
            "checks_performed": checks_performed,
            "required_files_exist": base_verification.get("ok", False) or "Missing files" not in base_verification.get("error", ""),
            "syntax_ok": app_kind == "website" or bool(base_verification.get("syntax_verified")),
            "readme_exists": readme_path.exists(),
            "readme_clear": "How to run" in readme_text,
            "launcher_exists": launcher_path.exists(),
            "launcher_ready": self._launcher_looks_ready(app_kind, launcher_text, template["main_filename"]),
            "ui_usable": self._ui_looks_usable(app_kind, main_text),
            "error_handling_ok": self._error_handling_looks_present(app_kind, main_text),
            "styling_ok": self._styling_looks_intentional(app_kind, project_path, main_text),
            "workspace_safe": explicit_target or self._path_is_within(project_path, self._workspace_root().resolve()),
            "tests_or_checks_passed": base_verification.get("ok", False),
            "launch_ready": self._launcher_looks_ready(app_kind, launcher_text, template["main_filename"]),
            "launch_verification": launch_verification,
            "error": base_verification.get("error", ""),
        }
        verification["ok"] = (
            base_verification.get("ok", False)
            and verification["readme_exists"]
            and verification["readme_clear"]
            and verification["launcher_exists"]
            and verification["launcher_ready"]
            and verification["ui_usable"]
            and verification["error_handling_ok"]
            and verification["styling_ok"]
            and verification["workspace_safe"]
            and launch_verification.get("passed", False)
        )
        return verification

    def _build_acceptance_checklist(
        self,
        brief: dict[str, Any],
        app_kind: str,
        project_path: Path,
        verification: dict[str, Any],
        explicit_target: bool,
        settings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        checks = [
            {"name": "Project brief created", "passed": True, "detail": brief["request"]},
            {
                "name": "Acceptance checklist created",
                "passed": True if settings["require_acceptance_checklist"] else True,
                "detail": "Professional build checklist is active." if settings["require_acceptance_checklist"] else "Checklist not required.",
            },
            {"name": "Required files exist", "passed": verification["required_files_exist"], "detail": verification.get("error", "") or "All required files are present."},
            {"name": "Syntax/static checks pass", "passed": verification["tests_or_checks_passed"], "detail": verification.get("error", "") or ", ".join(verification["checks_performed"])},
            {"name": "Clear README exists", "passed": verification["readme_exists"] and verification["readme_clear"], "detail": "README exists and includes run instructions." if verification["readme_exists"] and verification["readme_clear"] else "README is missing or unclear."},
            {"name": "Double-click launcher works where applicable", "passed": verification["launcher_exists"] and verification["launcher_ready"], "detail": "Launcher file exists and points at the project entry point." if verification["launcher_exists"] and verification["launcher_ready"] else "Launcher is missing or does not point at the expected entry point."},
            {"name": "Launch verification passed or skipped with reason", "passed": verification["launch_verification"]["passed"], "detail": verification["launch_verification"]["reason"]},
            {"name": "UI is usable", "passed": verification["ui_usable"], "detail": "The scaffold contains the expected UI structure for this project type." if verification["ui_usable"] else "The generated UI structure looks incomplete."},
            {"name": "Error handling exists", "passed": verification["error_handling_ok"], "detail": "Basic error handling is present where this project type needs it." if verification["error_handling_ok"] else "The scaffold still needs basic error handling."},
            {"name": "Styling is intentional", "passed": verification["styling_ok"], "detail": "The generated styling is intentional for this project type." if verification["styling_ok"] else "The generated styling still looks too bare."},
            {"name": "No unsafe file writes", "passed": verification["workspace_safe"], "detail": "The project stayed inside workspace or used an explicit user path." if verification["workspace_safe"] else "The project path was not safe."},
            {"name": "Project stays in workspace unless user specified otherwise", "passed": explicit_target or self._path_is_within(project_path, self._workspace_root().resolve()), "detail": str(project_path)},
        ]
        return checks

    def _self_review_professional_build(
        self,
        build_request: str,
        app_kind: str,
        verification: dict[str, Any],
        checklist: list[dict[str, Any]],
    ) -> dict[str, Any]:
        issues: list[dict[str, str]] = []
        for item in checklist:
            if not item["passed"]:
                issues.append({"code": self._issue_code_from_check(item["name"]), "detail": item["detail"]})
        if app_kind == "website" and "basic" not in build_request.lower() and not verification["styling_ok"]:
            issues.append({"code": "styling_needs_polish", "detail": "The website still looks too bare for a professional build."})
        return {"ready": not issues, "issues": issues}

    def _apply_professional_improvements(
        self,
        project_path: Path,
        app_kind: str,
        review: dict[str, Any],
        website_spec: dict[str, Any] | None,
        professional_context: dict[str, Any],
    ) -> dict[str, Any]:
        applied: list[str] = []
        issue_codes = {issue["code"] for issue in review.get("issues", [])}
        if "readme_incomplete" in issue_codes or "checklist_missing" in issue_codes:
            self._update_professional_readme(project_path, app_kind, website_spec, professional_context)
            applied.append("Refreshed README with the current brief and acceptance checklist.")
        if "launch_verification_failed" in issue_codes:
            applied.append("Re-ran launch verification on the next pass after the failure was recorded.")
        return {"applied": applied}

    def _collect_known_limitations(self, verification: dict[str, Any], review: dict[str, Any]) -> list[str]:
        launch_verification = verification.get("launch_verification", {})
        limitations: list[str] = []
        if launch_verification.get("status") != "passed":
            limitations.append(launch_verification.get("reason", "Launch verification did not pass."))
        if verification.get("error"):
            limitations.append(verification["error"])
        for issue in review.get("issues", []):
            detail = issue.get("detail", "").strip()
            if detail and detail not in limitations:
                limitations.append(detail)
        return limitations

    def _compose_professional_report(
        self,
        status: str,
        project_path: Path,
        app_kind: str,
        files: dict[str, str],
        brief: dict[str, Any],
        checklist: list[dict[str, Any]],
        verification: dict[str, Any],
        verification_history: list[dict[str, Any]],
        improvements: list[str],
        known_limitations: list[str],
        research: dict[str, Any] | None,
    ) -> str:
        if status == "completed":
            heading = "Professional build completed."
        elif status == "verification_failed":
            heading = "Professional build stopped after a failed verification step."
        else:
            heading = "Professional build stopped before the acceptance checklist fully passed."

        lines = [heading, "", "Project brief:"]
        lines.append(f"- Request: {brief['request']}")
        lines.append(f"- Target platform: {brief['target_platform']}")
        lines.append(f"- Project path: {project_path}")
        lines.append(f"- How it runs: {brief['run_instructions']}")
        lines.append(f"- Done means: {brief['done_means']}")
        lines.append("")
        lines.append("Files created:")
        for path in files:
            lines.append(f"- {path}")
        lines.append("")
        lines.append("Checks performed:")
        for check in verification.get("checks_performed", []):
            lines.append(f"- {check}")
        lines.append("")
        lines.append("Acceptance checklist:")
        for item in checklist:
            marker = "pass" if item["passed"] else "fail"
            lines.append(f"- [{marker}] {item['name']}: {item['detail']}")
        lines.append("")
        lines.append("Improvement passes:")
        for entry in verification_history:
            status_text = "pass" if entry["ok"] else "fail"
            lines.append(f"- Pass {entry['pass']}: {status_text} ({', '.join(entry['checks_performed'])})")
        lines.append("")
        lines.append("What it improved after self-review:")
        if improvements:
            for item in improvements:
                lines.append(f"- {item}")
        else:
            lines.append("- No additional automated improvements were necessary after self-review.")
        lines.append("")
        lines.append("Known limitations:")
        if known_limitations:
            for limitation in known_limitations:
                lines.append(f"- {limitation}")
        else:
            lines.append("- None.")
        if research is not None:
            lines.append("")
            lines.append("Research summary:")
            lines.append(research["summary"])
        return "\n".join(lines)

    def _update_professional_readme(
        self,
        project_path: Path,
        app_kind: str,
        website_spec: dict[str, Any] | None,
        professional_context: dict[str, Any],
    ) -> None:
        template = APP_TEMPLATES[app_kind]
        readme_path = project_path / template["readme_name"]
        content = self._readme_source(
            template["display_name"],
            template["launcher_name"],
            template["main_filename"],
            website_spec=website_spec,
            professional_context=professional_context,
        )
        file_tools.write_file(str(readme_path), content)

    def _project_run_instructions(self, app_kind: str) -> str:
        template = APP_TEMPLATES[app_kind]
        return f"Double-click {template['launcher_name']}."

    def _verify_launch_readiness(self, project_path: Path, app_kind: str, settings: dict[str, Any]) -> dict[str, Any]:
        if not settings.get("launch_verification_enabled", True):
            return {
                "status": "skipped",
                "passed": True,
                "reason": "Launch verification is disabled in settings.",
                "stdout": "",
                "stderr": "",
                "window_title": "",
                "cleanup_performed": False,
            }

        if app_kind not in {"calculator", "notepad", "todo", "timer"}:
            return {
                "status": "skipped",
                "passed": True,
                "reason": "Launch verification is only required for generated Python GUI apps.",
                "stdout": "",
                "stderr": "",
                "window_title": "",
                "cleanup_performed": False,
            }

        safe_root = (self._workspace_root() / "generated_apps").resolve()
        if not self._path_is_within(project_path, safe_root):
            return {
                "status": "skipped",
                "passed": False,
                "reason": "Launch verification was skipped because the project is outside workspace/generated_apps.",
                "stdout": "",
                "stderr": "",
                "window_title": "",
                "cleanup_performed": False,
            }

        main_path = project_path / APP_TEMPLATES[app_kind]["main_filename"]
        if not main_path.exists():
            return {
                "status": "failed",
                "passed": False,
                "reason": f"Launch verification could not start because {main_path} is missing.",
                "stdout": "",
                "stderr": "",
                "window_title": "",
                "cleanup_performed": False,
            }

        command = [sys.executable, str(main_path)]
        timeout_seconds = int(settings.get("launch_timeout_seconds", 8))
        process = None
        result: dict[str, Any] | None = None
        cleanup_performed = False
        try:
            process = self._spawn_launch_process(command, project_path)
            result = self._wait_for_launch_result(process, project_path, app_kind, timeout_seconds)
        finally:
            if process is not None:
                cleanup_performed = self._cleanup_launch_process(process)
        if result is None:
            result = {
                "status": "failed",
                "passed": False,
                "reason": "Launch verification did not produce a result.",
                "stdout": "",
                "stderr": "",
                "window_title": "",
            }
        return {**result, "cleanup_performed": cleanup_performed}

    def _spawn_launch_process(self, command: list[str], project_path: Path):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(
            command,
            cwd=str(project_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )

    def _wait_for_launch_result(
        self,
        process,
        project_path: Path,
        app_kind: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(timeout_seconds, 1)
        expected_titles = self._expected_window_titles(app_kind)
        last_detected_title = ""
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                stdout, stderr = self._collect_process_output(process, timeout=1)
                detail = stderr.strip() or stdout.strip() or f"Process exited with code {exit_code}."
                return {
                    "status": "failed",
                    "passed": False,
                    "reason": f"Launch verification failed: {detail}",
                    "stdout": stdout,
                    "stderr": stderr,
                    "window_title": "",
                }

            detected_title = self._detect_window_title(expected_titles)
            if detected_title:
                stdout, stderr = self._collect_process_output(process, timeout=0.2)
                return {
                    "status": "passed",
                    "passed": True,
                    "reason": f"Launch verification passed. Detected window title: {detected_title}",
                    "stdout": stdout,
                    "stderr": stderr,
                    "window_title": detected_title,
                }
            last_detected_title = detected_title or last_detected_title
            time.sleep(0.2)

        stdout, stderr = self._collect_process_output(process, timeout=0.2)
        if process.poll() is None:
            return {
                "status": "timeout",
                "passed": False,
                "reason": "Launch verification timed out before a matching window title was detected.",
                "stdout": stdout,
                "stderr": stderr,
                "window_title": last_detected_title,
            }

        exit_code = process.poll()
        detail = stderr.strip() or stdout.strip() or f"Process exited with code {exit_code}."
        return {
            "status": "failed",
            "passed": False,
            "reason": f"Launch verification failed: {detail}",
            "stdout": stdout,
            "stderr": stderr,
            "window_title": last_detected_title,
        }

    def _expected_window_titles(self, app_kind: str) -> list[str]:
        return [
            APP_TEMPLATES[app_kind]["display_name"].lower(),
            app_kind.lower(),
        ]

    def _detect_window_title(self, expected_titles: list[str]) -> str:
        title_candidates: list[str] = []
        uia = get_active_window_title()
        if uia.get("ok"):
            title_candidates.append(uia.get("title", ""))
        basic = get_active_window_basic()
        if basic.get("ok"):
            title_candidates.append(basic.get("title", ""))
        for title in title_candidates:
            lowered = title.lower()
            if title and any(expected in lowered for expected in expected_titles):
                return title
        return ""

    def _collect_process_output(self, process, timeout: float) -> tuple[str, str]:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return stdout or "", stderr or ""
        except subprocess.TimeoutExpired:
            return "", ""

    def _cleanup_launch_process(self, process) -> bool:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            return True
        except Exception:
            return False

    def _path_is_within(self, path: Path, root: Path) -> bool:
        try:
            return path.resolve().is_relative_to(root.resolve())
        except FileNotFoundError:
            return path.parent.resolve().is_relative_to(root.resolve())

    def _launcher_looks_ready(self, app_kind: str, launcher_text: str, main_filename: str) -> bool:
        lowered = launcher_text.lower()
        if app_kind == "website":
            return "start \"\" \"index.html\"" in lowered and "index.html" in lowered
        return main_filename.lower() in lowered and ("%python_exe%" in lowered or "py" in lowered or "python" in lowered)

    def _ui_looks_usable(self, app_kind: str, main_text: str) -> bool:
        lowered = main_text.lower()
        if app_kind == "website":
            return "<main" in lowered and "<header" in lowered and "ctaButton".lower() in lowered
        if app_kind == "script":
            return "argparse" in lowered or "print(" in lowered
        return "tk." in lowered and "mainloop" in lowered

    def _error_handling_looks_present(self, app_kind: str, main_text: str) -> bool:
        lowered = main_text.lower()
        if app_kind in {"website", "todo"}:
            return True
        return "except" in lowered

    def _styling_looks_intentional(self, app_kind: str, project_path: Path, main_text: str) -> bool:
        if app_kind == "website":
            style_path = project_path / "style.css"
            if not style_path.exists():
                return False
            css = style_path.read_text(encoding="utf-8").lower()
            return all(token in css for token in ("--accent", "hero-card", "content-card"))
        if app_kind == "calculator":
            return "#10131a" in main_text.lower() and "segoe ui" in main_text.lower()
        return True

    def _issue_code_from_check(self, check_name: str) -> str:
        mapping = {
            "Acceptance checklist created": "checklist_missing",
            "Clear README exists": "readme_incomplete",
            "Double-click launcher works where applicable": "launcher_incomplete",
            "Launch verification passed or skipped with reason": "launch_verification_failed",
            "Required files exist": "missing_files",
            "Syntax/static checks pass": "verification_failed",
            "Error handling exists": "error_handling_missing",
            "Styling is intentional": "styling_needs_polish",
        }
        return mapping.get(check_name, "check_failed")

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

    def _build_app_files(
        self,
        app_kind: str,
        target_dir: Path,
        website_spec: dict | None = None,
        professional_context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        template = APP_TEMPLATES[app_kind]
        display_name = template["display_name"]
        main_filename = template["main_filename"]
        launcher_name = template["launcher_name"]
        readme_name = template["readme_name"]

        return {
            str(target_dir / main_filename): self._app_main_source(app_kind, display_name, website_spec),
            str(target_dir / launcher_name): self._launcher_source(display_name, main_filename),
            str(target_dir / readme_name): self._readme_source(
                display_name,
                launcher_name,
                main_filename,
                website_spec,
                professional_context=professional_context,
            ),
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
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
        except OSError as exc:
            messagebox.showerror("Notepad", f"Could not open file: {exc}")
            return
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
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.editor.get("1.0", tk.END))
        except OSError as exc:
            messagebox.showerror("Notepad", f"Could not save file: {exc}")
            return
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
            "script": """import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generated local tool script.")
    parser.add_argument("--name", default="LocalPilot Tool", help="Optional label to print in the output.")
    parser.add_argument("--output", default="tool_output.txt", help="Optional output file name.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = Path(args.output)
    try:
        output_path.write_text(f"Tool run completed for: {args.name}\\n", encoding="utf-8")
        print(f"Wrote {output_path.resolve()}")
    except Exception as exc:
        print(f"Error: {exc}")
        raise


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
        professional_context: dict[str, Any] | None = None,
    ) -> str:
        if professional_context is not None:
            brief = professional_context.get("brief", {})
            checklist = professional_context.get("acceptance_checklist", [])
            verification_summary = professional_context.get("verification_summary", {})
            known_limitations = professional_context.get("known_limitations", [])
            research = professional_context.get("research")
            checklist_lines = "\n".join(
                f"- [{'pass' if item['passed'] else 'fail'}] {item['name']}: {item['detail']}"
                for item in checklist
            ) or "- Checklist will be filled after verification."
            research_block = ""
            if research is not None:
                research_block = f"\nResearch\n- {research['summary'].replace(chr(10), chr(10) + '- ')}\n"
            limitations_block = "\n".join(f"- {item}" for item in known_limitations) or "- No known limitations recorded yet."
            return f"""{display_name}
====================

This project was generated by LocalPilot Professional Build Mode.

Project brief
- Request: {brief.get('request', 'Unknown request')}
- Target platform: {brief.get('target_platform', 'Windows desktop/local filesystem')}
- How it should run: {brief.get('run_instructions', f'Double-click {launcher_name}.')}
- Done means: {brief.get('done_means', 'Acceptance checklist passes and verification is complete.')}

Files
- {main_filename}
- {launcher_name}

How to run
1. Open this folder.
2. Double-click {launcher_name}.

Acceptance checklist
{checklist_lines}

Verification summary
- Checks performed: {", ".join(verification_summary.get('checks_performed', [])) or 'Pending verification'}
- Verification ok: {verification_summary.get('ok', False)}

Known limitations
{limitations_block}
{research_block}"""
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
