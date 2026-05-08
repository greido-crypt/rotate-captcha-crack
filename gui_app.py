"""
gui_app.py — customtkinter GUI for rotate-captcha-crack Neural API Server.
Entry point for the packaged EXE.
"""

import os
import queue
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import customtkinter as ctk
import torch

from server_core import ServerManager

# ── Runtime path: chdir so ./models resolves correctly inside the EXE ─────────
if getattr(sys, "frozen", False):
    os.chdir(Path(sys.executable).parent)

# ── Theme ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Palette ───────────────────────────────────────────────────────────────────
CLR_BG        = "#1a1b1e"
CLR_PANEL     = "#25262b"
CLR_CARD      = "#2c2d32"
CLR_BORDER    = "#373a40"
CLR_GREEN     = "#2dd4bf"
CLR_RED       = "#f87171"
CLR_YELLOW    = "#fbbf24"
CLR_BLUE      = "#60a5fa"
CLR_TEXT      = "#e5e7eb"
CLR_MUTED     = "#9ca3af"
CLR_BTN_START = "#16a34a"
CLR_BTN_STOP  = "#dc2626"
CLR_BTN_HOV_S = "#15803d"
CLR_BTN_HOV_X = "#b91c1c"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self._mgr = ServerManager()
        self._log_q: queue.Queue = queue.Queue()
        self._mgr.set_log_callback(self._enqueue_log)

        self._start_time: float | None = None
        self._server_thread: threading.Thread | None = None

        # ── Window ────────────────────────────────────────────────────────────
        self.title("rotate-captcha-crack  |  Neural API Server")
        self.geometry("960x620")
        self.minsize(800, 540)
        self.configure(fg_color=CLR_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh_gpu_info()
        self._tick()   # start polling loop

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Left sidebar ──────────────────────────────────────────────────────
        sidebar = ctk.CTkFrame(self, width=260, fg_color=CLR_PANEL, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)   # ← items stretch to full width
        sidebar.grid_rowconfigure(99, weight=1)     # ← spacer before stats block
        sidebar.grid_propagate(False)

        row = 0

        # Logo / title
        ctk.CTkLabel(
            sidebar, text="⚡ Neural API Server",
            font=ctk.CTkFont(size=15, weight="bold"), text_color=CLR_TEXT
        ).grid(row=row, column=0, padx=16, pady=(14, 2), sticky="w"); row += 1

        ctk.CTkLabel(
            sidebar, text="rotate-captcha-crack",
            font=ctk.CTkFont(size=10), text_color=CLR_MUTED
        ).grid(row=row, column=0, padx=16, pady=(0, 10), sticky="w"); row += 1

        self._sep(sidebar, row); row += 1

        # Status indicator
        ctk.CTkLabel(
            sidebar, text="СТАТУС", font=ctk.CTkFont(size=9, weight="bold"),
            text_color=CLR_MUTED
        ).grid(row=row, column=0, padx=16, pady=(8, 2), sticky="w"); row += 1

        status_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        status_row.grid(row=row, column=0, padx=16, pady=(0, 8), sticky="w"); row += 1

        self._dot = ctk.CTkLabel(status_row, text="●", font=ctk.CTkFont(size=16),
                                  text_color=CLR_RED)
        self._dot.pack(side="left", padx=(0, 6))
        self._status_lbl = ctk.CTkLabel(status_row, text="Остановлен",
                                         font=ctk.CTkFont(size=12), text_color=CLR_TEXT)
        self._status_lbl.pack(side="left")

        self._sep(sidebar, row); row += 1

        # Mode selection
        ctk.CTkLabel(
            sidebar, text="РЕЖИМ", font=ctk.CTkFont(size=9, weight="bold"),
            text_color=CLR_MUTED
        ).grid(row=row, column=0, padx=16, pady=(8, 2), sticky="w"); row += 1

        self._mode_var = ctk.StringVar(value="gpu")
        modes = [("GPU  FP16  (быстрый)", "gpu"),
                 ("CPU  FP32  (без GPU)", "cpu"),
                 ("CPU  INT8  (квантизация)", "quant")]
        for label, val in modes:
            ctk.CTkRadioButton(
                sidebar, text=label, variable=self._mode_var, value=val,
                font=ctk.CTkFont(size=11), text_color=CLR_TEXT,
                fg_color=CLR_BLUE, hover_color=CLR_BLUE,
            ).grid(row=row, column=0, padx=20, pady=2, sticky="w"); row += 1

        if not torch.cuda.is_available():
            self._mode_var.set("cpu")

        self._sep(sidebar, row); row += 1

        # Port
        ctk.CTkLabel(
            sidebar, text="ПОРТ", font=ctk.CTkFont(size=9, weight="bold"),
            text_color=CLR_MUTED
        ).grid(row=row, column=0, padx=16, pady=(8, 2), sticky="w"); row += 1

        self._port_var = ctk.StringVar(value="4396")
        ctk.CTkEntry(
            sidebar, textvariable=self._port_var, width=110,
            font=ctk.CTkFont(size=12), fg_color=CLR_CARD, border_color=CLR_BORDER,
        ).grid(row=row, column=0, padx=16, pady=(0, 8), sticky="w"); row += 1

        self._sep(sidebar, row); row += 1

        # Start / Stop button
        self._btn = ctk.CTkButton(
            sidebar, text="▶   Запустить",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40, corner_radius=8,
            fg_color=CLR_BTN_START, hover_color=CLR_BTN_HOV_S,
            command=self._toggle,
        )
        self._btn.grid(row=row, column=0, padx=16, pady=12, sticky="ew"); row += 1

        # ── Spacer: pushes stats block to bottom ──
        ctk.CTkLabel(sidebar, text="").grid(row=99, column=0)

        self._sep(sidebar, 100); row = 101

        # Stats
        ctk.CTkLabel(
            sidebar, text="СТАТИСТИКА", font=ctk.CTkFont(size=9, weight="bold"),
            text_color=CLR_MUTED
        ).grid(row=row, column=0, padx=16, pady=(8, 2), sticky="w"); row += 1

        self._req_lbl    = self._stat_row(sidebar, row, "Запросов:", "0"); row += 1
        self._uptime_lbl = self._stat_row(sidebar, row, "Аптайм:",   "—"); row += 1

        self._sep(sidebar, row); row += 1

        # GPU info
        ctk.CTkLabel(
            sidebar, text="СИСТЕМА", font=ctk.CTkFont(size=9, weight="bold"),
            text_color=CLR_MUTED
        ).grid(row=row, column=0, padx=16, pady=(8, 2), sticky="w"); row += 1

        self._gpu_name_lbl = self._stat_row(sidebar, row, "GPU:",  "Нет"); row += 1
        self._gpu_vram_lbl = self._stat_row(sidebar, row, "VRAM:", "—");   row += 1

        ctk.CTkLabel(sidebar, text="").grid(row=row, column=0, pady=6)

        # ── Right panel — logs ────────────────────────────────────────────────
        right = ctk.CTkFrame(self, fg_color=CLR_BG, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(1, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Log header
        hdr = ctk.CTkFrame(right, fg_color=CLR_PANEL, height=44, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)

        ctk.CTkLabel(
            hdr, text="📋  Журнал сервера",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=CLR_TEXT
        ).pack(side="left", padx=16, pady=10)

        ctk.CTkButton(
            hdr, text="Очистить", width=80, height=28,
            font=ctk.CTkFont(size=11), fg_color=CLR_CARD, hover_color=CLR_BORDER,
            command=self._clear_log,
        ).pack(side="right", padx=12, pady=8)

        # URL hint (hidden until server starts)
        self._url_lbl = ctk.CTkLabel(
            hdr, text="", font=ctk.CTkFont(size=11), text_color=CLR_GREEN
        )
        self._url_lbl.pack(side="right", padx=4)

        # Log textbox
        self._log_box = ctk.CTkTextbox(
            right, font=ctk.CTkFont(family="Courier New", size=12),
            fg_color=CLR_CARD, text_color=CLR_TEXT,
            corner_radius=0, border_width=0,
            wrap="word", state="disabled",
        )
        self._log_box.grid(row=1, column=0, sticky="nsew")

        # Bottom status bar
        bar = ctk.CTkFrame(right, fg_color=CLR_PANEL, height=28, corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)

        self._bar_lbl = ctk.CTkLabel(
            bar, text="Сервер не запущен",
            font=ctk.CTkFont(size=11), text_color=CLR_MUTED
        )
        self._bar_lbl.pack(side="left", padx=12)

    # ══════════════════════════════════════════════════════════════════════════
    # Logic
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle(self):
        if self._mgr.is_running:
            self._do_stop()
        else:
            self._do_start()

    def _do_start(self):
        mode = self._mode_var.get()
        try:
            port = int(self._port_var.get())
        except ValueError:
            self._append_log("ERROR: некорректный порт")
            return

        self._set_state("loading")

        def _worker():
            try:
                self._mgr.start(mode=mode, port=port)
                self._start_time = time.time()
                self.after(0, lambda: self._set_state("running", port))
            except Exception as exc:
                self._enqueue_log(f"ERROR: {exc}")
                self.after(0, lambda: self._set_state("stopped"))

        self._server_thread = threading.Thread(target=_worker, daemon=True)
        self._server_thread.start()

    def _do_stop(self):
        self._set_state("loading")

        def _worker():
            self._mgr.stop()
            self._start_time = None
            self.after(0, lambda: self._set_state("stopped"))

        threading.Thread(target=_worker, daemon=True).start()

    def _set_state(self, state: str, port: int | None = None):
        if state == "running":
            self._dot.configure(text_color=CLR_GREEN)
            self._status_lbl.configure(text="Работает")
            self._btn.configure(text="■   Остановить",
                                 fg_color=CLR_BTN_STOP, hover_color=CLR_BTN_HOV_X)
            self._url_lbl.configure(text=f"http://127.0.0.1:{port}/docs")
            self._bar_lbl.configure(text=f"Слушает порт {port}  |  /docs для тестирования")
        elif state == "loading":
            self._dot.configure(text_color=CLR_YELLOW)
            self._status_lbl.configure(text="Загрузка…")
            self._btn.configure(state="disabled")
        else:  # stopped
            self._dot.configure(text_color=CLR_RED)
            self._status_lbl.configure(text="Остановлен")
            self._btn.configure(text="▶   Запустить", state="normal",
                                 fg_color=CLR_BTN_START, hover_color=CLR_BTN_HOV_S)
            self._url_lbl.configure(text="")
            self._bar_lbl.configure(text="Сервер не запущен")

    # ── Polling loop (every 500ms) ────────────────────────────────────────────

    def _tick(self):
        # Drain log queue
        try:
            while True:
                self._append_log(self._log_q.get_nowait())
        except queue.Empty:
            pass

        # Detect unexpected server crash (was running, now not)
        if getattr(self, "_was_running", False) and not self._mgr.is_running:
            self._set_state("stopped")
            self._append_log("[WARN]  Server stopped unexpectedly")
        self._was_running = self._mgr.is_running

        # Update stats
        if self._mgr.is_running:
            self._req_lbl.configure(text=str(self._mgr.request_count))
            if self._start_time:
                elapsed = int(time.time() - self._start_time)
                self._uptime_lbl.configure(text=str(timedelta(seconds=elapsed)))

            # GPU VRAM live update
            info = ServerManager.gpu_info()
            if info["available"]:
                self._gpu_vram_lbl.configure(
                    text=f"{info['used_gb']:.1f} / {info['total_gb']} GB"
                )

        self.after(500, self._tick)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enqueue_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_q.put(f"[{ts}]  {msg}")

    def _append_log(self, msg: str):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _clear_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _refresh_gpu_info(self):
        info = ServerManager.gpu_info()
        if info["available"]:
            self._gpu_name_lbl.configure(text=info["name"])
            self._gpu_vram_lbl.configure(text=f"— / {info['total_gb']} GB")
        else:
            self._gpu_name_lbl.configure(text="Нет NVIDIA GPU")
            self._gpu_vram_lbl.configure(text="—")

    def _on_close(self):
        if self._mgr.is_running:
            threading.Thread(target=self._mgr.stop, daemon=True).start()
        self.destroy()

    @staticmethod
    def _sep(parent, row):
        ctk.CTkFrame(parent, height=1, fg_color=CLR_BORDER).grid(
            row=row, column=0, sticky="ew", padx=12, pady=2
        )

    @staticmethod
    def _stat_row(parent, row, label: str, value: str) -> ctk.CTkLabel:
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=0, padx=16, pady=1, sticky="ew")
        f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=11),
                     text_color=CLR_MUTED, width=72, anchor="w").grid(row=0, column=0, sticky="w")
        lbl = ctk.CTkLabel(f, text=value, font=ctk.CTkFont(size=11),
                           text_color=CLR_TEXT, anchor="w")
        lbl.grid(row=0, column=1, sticky="w")
        return lbl


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    App().mainloop()
