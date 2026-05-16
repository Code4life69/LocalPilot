from __future__ import annotations

import argparse
import sys
import tkinter as tk
from tkinter import ttk


class SafeClickTestWindow:
    def __init__(self, auto_close_after: int | None = None, topmost: bool = True) -> None:
        self.auto_close_after = auto_close_after if auto_close_after is None else max(int(auto_close_after), 1)
        self.root = tk.Tk()
        self.root.title("LocalPilot Safe Click Test")
        self.root.geometry("760x360+220+180")
        self.root.configure(bg="#08131d")
        if topmost:
            self.root.attributes("-topmost", True)
        self.click_count = 0
        self.count_var = tk.StringVar(value="Button clicks: 0")
        self.status_var = tk.StringVar(value="Ready for one safe desktop click.")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        frame = tk.Frame(self.root, bg="#08131d", padx=24, pady=24)
        frame.pack(fill="both", expand=True)

        title = tk.Label(
            frame,
            text="LocalPilot Safe Click Test",
            font=("Segoe UI", 22, "bold"),
            fg="#edf6ff",
            bg="#08131d",
        )
        title.pack(pady=(0, 18))

        button = ttk.Button(
            frame,
            text="SAFE TEST BUTTON",
            command=self.on_click,
        )
        button.pack(fill="x", ipadx=24, ipady=26, pady=(0, 20))

        style = ttk.Style()
        style.configure("SafeClick.TButton", font=("Segoe UI", 24, "bold"))
        button.configure(style="SafeClick.TButton")

        count_label = tk.Label(
            frame,
            textvariable=self.count_var,
            font=("Segoe UI", 18, "bold"),
            fg="#8ef1be",
            bg="#08131d",
        )
        count_label.pack(pady=(0, 10))

        status_label = tk.Label(
            frame,
            textvariable=self.status_var,
            font=("Segoe UI", 12),
            fg="#b9d3e7",
            bg="#08131d",
            wraplength=660,
            justify="center",
        )
        status_label.pack()

    def on_click(self) -> None:
        self.click_count += 1
        self.count_var.set(f"Button clicks: {self.click_count}")
        self.status_var.set("SAFE TEST BUTTON clicked successfully.")
        print(f"SAFE_TEST_BUTTON_CLICK_COUNT: {self.click_count}", flush=True)
        if self.auto_close_after is not None and self.click_count >= self.auto_close_after:
            self.root.after(250, self.close)

    def run(self) -> int:
        print("SAFE_CLICK_TEST_WINDOW_READY", flush=True)
        self.root.lift()
        self.root.focus_force()
        self.root.mainloop()
        return 0

    def close(self) -> None:
        print("SAFE_CLICK_TEST_WINDOW_CLOSED", flush=True)
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.desktop_click_test_window")
    parser.add_argument("--auto-close-after", type=int, default=None, help="Close automatically after N successful button clicks.")
    parser.add_argument("--not-topmost", action="store_true", help="Do not keep the window topmost.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    window = SafeClickTestWindow(auto_close_after=args.auto_close_after, topmost=not args.not_topmost)
    return window.run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
