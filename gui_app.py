"""
gui_app.py — customtkinter GUI for rotate-captcha-crack Neural API Server.
"""

import os
import queue
import sys
import threading
import time
import webbrowser
from datetime import timedelta
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import psutil
import torch

from server_core import ServerManager

# ── Runtime path ──────────────────────────────────────────────────────────────
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
        self._current_port: int = 4396
        self._cpu_sampler = psutil.Process(os.getpid())

        self.title("rotate-captcha-crack  |  Neural API Server")
        self.geometry("980x640")
        self.minsize(820, 560)
        self.configure(fg_color=CLR_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh_gpu_info()
        self._tick()

    # ══════════════════════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ───────────────────────────────────────────────────────────
        sb = ctk.CTkFrame(self, width=265, fg_color=CLR_PANEL, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(99, weight=1)
        sb.grid_propagate(False)

        r = 0

        # Title
        ctk.CTkLabel(sb, text="⚡ Neural API Server",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=CLR_TEXT
        ).grid(row=r, column=0, padx=16, pady=(14, 2), sticky="w"); r += 1

        ctk.CTkLabel(sb, text="rotate-captcha-crack",
                     font=ctk.CTkFont(size=10), text_color=CLR_MUTED
        ).grid(row=r, column=0, padx=16, pady=(0, 10), sticky="w"); r += 1

        self._sep(sb, r); r += 1

        # Status
        self._section(sb, r, "СТАТУС"); r += 1
        status_row = ctk.CTkFrame(sb, fg_color="transparent")
        status_row.grid(row=r, column=0, padx=16, pady=(0, 8), sticky="w"); r += 1
        self._dot = ctk.CTkLabel(status_row, text="●",
                                  font=ctk.CTkFont(size=16), text_color=CLR_RED)
        self._dot.pack(side="left", padx=(0, 6))
        self._status_lbl = ctk.CTkLabel(status_row, text="Остановлен",
                                         font=ctk.CTkFont(size=12), text_color=CLR_TEXT)
        self._status_lbl.pack(side="left")

        self._sep(sb, r); r += 1

        # Mode
        self._section(sb, r, "РЕЖИМ"); r += 1
        self._mode_var = ctk.StringVar(value="gpu" if torch.cuda.is_available() else "cpu")
        for label, val in [("GPU  FP16  (быстрый)", "gpu"), ("CPU  FP32  (без GPU)", "cpu")]:
            ctk.CTkRadioButton(sb, text=label, variable=self._mode_var, value=val,
                               font=ctk.CTkFont(size=11), text_color=CLR_TEXT,
                               fg_color=CLR_BLUE, hover_color=CLR_BLUE,
            ).grid(row=r, column=0, padx=20, pady=2, sticky="w"); r += 1

        self._sep(sb, r); r += 1

        # Port
        self._section(sb, r, "ПОРТ"); r += 1
        self._port_var = ctk.StringVar(value="4396")
        ctk.CTkEntry(sb, textvariable=self._port_var, width=110,
                     font=ctk.CTkFont(size=12),
                     fg_color=CLR_CARD, border_color=CLR_BORDER,
        ).grid(row=r, column=0, padx=16, pady=(0, 8), sticky="w"); r += 1

        self._sep(sb, r); r += 1

        # Save images toggle + dir
        self._section(sb, r, "СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ"); r += 1
        self._save_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sb, text="Сохранять пары", variable=self._save_var,
                      font=ctk.CTkFont(size=11), text_color=CLR_TEXT,
                      progress_color=CLR_BLUE,
                      command=self._on_save_toggle,
        ).grid(row=r, column=0, padx=16, pady=(0, 4), sticky="w"); r += 1

        self._save_dir_lbl = ctk.CTkLabel(sb, text="Папка не выбрана",
                                          font=ctk.CTkFont(size=10),
                                          text_color=CLR_MUTED, wraplength=220, anchor="w")
        self._save_dir_lbl.grid(row=r, column=0, padx=16, pady=(0, 2), sticky="w"); r += 1

        ctk.CTkButton(sb, text="Выбрать папку…", height=28,
                      font=ctk.CTkFont(size=11),
                      fg_color=CLR_CARD, hover_color=CLR_BORDER,
                      command=self._pick_save_dir,
        ).grid(row=r, column=0, padx=16, pady=(0, 8), sticky="ew"); r += 1

        self._sep(sb, r); r += 1

        # Start/Stop
        self._btn = ctk.CTkButton(sb, text="▶   Запустить",
                                   font=ctk.CTkFont(size=13, weight="bold"),
                                   height=40, corner_radius=8,
                                   fg_color=CLR_BTN_START, hover_color=CLR_BTN_HOV_S,
                                   command=self._toggle)
        self._btn.grid(row=r, column=0, padx=16, pady=12, sticky="ew"); r += 1

        # ── Spacer ────────────────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="").grid(row=99, column=0)
        self._sep(sb, 100); r = 101

        # Stats
        self._section(sb, r, "СТАТИСТИКА"); r += 1
        self._req_lbl    = self._stat_row(sb, r, "Запросов:", "0");   r += 1
        self._uptime_lbl = self._stat_row(sb, r, "Аптайм:",   "—");   r += 1
        self._cpu_lbl    = self._stat_row(sb, r, "CPU:",      "—");   r += 1

        self._sep(sb, r); r += 1

        # System
        self._section(sb, r, "СИСТЕМА"); r += 1
        self._gpu_name_lbl = self._stat_row(sb, r, "GPU:",  "Нет"); r += 1
        self._gpu_vram_lbl = self._stat_row(sb, r, "VRAM:", "—");   r += 1
        ctk.CTkLabel(sb, text="").grid(row=r, column=0, pady=6)

        # ── Right panel ───────────────────────────────────────────────────────
        right = ctk.CTkFrame(self, fg_color=CLR_BG, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(1, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Header
        hdr = ctk.CTkFrame(right, fg_color=CLR_PANEL, height=44, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(hdr, text="📋  Журнал сервера",
                     font=ctk.CTkFont(size=13, weight="bold"), text_color=CLR_TEXT
        ).grid(row=0, column=0, padx=14, pady=10, sticky="w")

        # Clickable docs URL
        self._docs_btn = ctk.CTkButton(
            hdr, text="", width=0,
            font=ctk.CTkFont(size=11), text_color=CLR_GREEN,
            fg_color="transparent", hover_color=CLR_PANEL,
            command=self._open_docs,
        )
        self._docs_btn.grid(row=0, column=1, padx=4, sticky="w")

        ctk.CTkButton(hdr, text="Очистить", width=80, height=28,
                      font=ctk.CTkFont(size=11),
                      fg_color=CLR_CARD, hover_color=CLR_BORDER,
                      command=self._clear_log,
        ).grid(row=0, column=2, padx=12, pady=8)

        # Log textbox
        self._log_box = ctk.CTkTextbox(
            right, font=ctk.CTkFont(family="Courier New", size=12),
            fg_color=CLR_CARD, text_color=CLR_TEXT,
            corner_radius=0, border_width=0,
            wrap="word", state="disabled",
        )
        self._log_box.grid(row=1, column=0, sticky="nsew")

        # Status bar
        bar = ctk.CTkFrame(right, fg_color=CLR_PANEL, height=28, corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)
        self._bar_lbl = ctk.CTkLabel(bar, text="Сервер не запущен",
                                      font=ctk.CTkFont(size=11), text_color=CLR_MUTED)
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

        self._current_port = port
        self._set_state("loading")

        # Apply save_dir setting
        if self._save_var.get() and self._mgr.save_dir:
            pass  # already set
        else:
            self._mgr.save_dir = None

        def _worker():
            try:
                self._mgr.start(mode=mode, port=port)
                self._start_time = time.time()
                self.after(0, lambda: self._set_state("running", port))
            except Exception as exc:
                self._enqueue_log(f"ERROR: {exc}")
                self.after(0, lambda: self._set_state("stopped"))

        threading.Thread(target=_worker, daemon=True).start()

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
            self._btn.configure(text="■   Остановить", state="normal",
                                 fg_color=CLR_BTN_STOP, hover_color=CLR_BTN_HOV_X)
            url = f"http://127.0.0.1:{port}/docs"
            self._docs_btn.configure(text=f"🔗 {url}")
            self._bar_lbl.configure(text=f"Слушает порт {port}  |  нажмите ссылку для /docs")
        elif state == "loading":
            self._dot.configure(text_color=CLR_YELLOW)
            self._status_lbl.configure(text="Загрузка…")
            self._btn.configure(state="disabled")
        else:
            self._dot.configure(text_color=CLR_RED)
            self._status_lbl.configure(text="Остановлен")
            self._btn.configure(text="▶   Запустить", state="normal",
                                 fg_color=CLR_BTN_START, hover_color=CLR_BTN_HOV_S)
            self._docs_btn.configure(text="")
            self._bar_lbl.configure(text="Сервер не запущен")

    def _open_docs(self):
        if self._mgr.is_running:
            webbrowser.open(f"http://127.0.0.1:{self._current_port}/docs")

    # ── Image saving ──────────────────────────────────────────────────────────

    def _on_save_toggle(self):
        if self._save_var.get() and not self._mgr.save_dir:
            self._pick_save_dir()
        if not self._save_var.get():
            self._mgr.save_dir = None

    def _pick_save_dir(self):
        d = filedialog.askdirectory(title="Выберите папку для сохранения изображений")
        if d:
            self._mgr.save_dir = Path(d)
            short = Path(d).name
            self._save_dir_lbl.configure(text=f"…/{short}", text_color=CLR_GREEN)
            self._save_var.set(True)

    # ── Polling loop ─────────────────────────────────────────────────────────

    def _tick(self):
        # Drain log queue
        try:
            while True:
                self._append_log(self._log_q.get_nowait())
        except queue.Empty:
            pass

        if self._mgr.is_running:
            # Requests + uptime
            self._req_lbl.configure(text=str(self._mgr.request_count))
            if self._start_time:
                self._uptime_lbl.configure(
                    text=str(timedelta(seconds=int(time.time() - self._start_time)))
                )
            # CPU %
            cpu = psutil.cpu_percent(interval=None)
            self._cpu_lbl.configure(text=f"{cpu:.0f}%")
            # GPU VRAM
            info = ServerManager.gpu_info()
            if info["available"]:
                self._gpu_vram_lbl.configure(
                    text=f"{info['used_gb']:.1f} / {info['total_gb']} GB"
                )
        else:
            self._cpu_lbl.configure(text="—")

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
            # shorten e.g. "NVIDIA GeForce RTX 4050 Laptop GPU" → "RTX 4050 Laptop"
            name = (info["name"]
                    .replace("NVIDIA GeForce ", "")
                    .replace("NVIDIA ", "")
                    .replace(" GPU", "")
                    .strip())
            self._gpu_name_lbl.configure(text=name)
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
    def _section(parent, row, text: str):
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=CLR_MUTED
        ).grid(row=row, column=0, padx=16, pady=(8, 2), sticky="w")

    @staticmethod
    def _stat_row(parent, row, label: str, value: str) -> ctk.CTkLabel:
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=0, padx=16, pady=1, sticky="ew")
        f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=11),
                     text_color=CLR_MUTED, width=72, anchor="w"
        ).grid(row=0, column=0, sticky="w")
        lbl = ctk.CTkLabel(f, text=value, font=ctk.CTkFont(size=11),
                           text_color=CLR_TEXT, anchor="w")
        lbl.grid(row=0, column=1, sticky="w")
        return lbl


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    App().mainloop()
