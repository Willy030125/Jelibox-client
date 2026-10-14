"""
Export Dataset window, in the style of Roboflow's "Create New Version":

  1. Split          use the splits assigned in Jelibox, or cut by percentage with a slider (image counts shown)
  2. Preprocessing  Auto-Orient (discard EXIF rotation) and Resize
  3. Augmentation   a gallery of augmentations; a click opens one with a live preview of a random image of the
                    dataset and its sliders. Augmented copies are made for the Train split only.

The work itself is in dataset_export.py / augmentation.py / export_preprocess.py; this file is only the window.
"""
import os
import random
import shutil
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import augmentation
from . import config as _config
from . import dataset_export
from . import splits as image_splits
from .augment_geometry import RESIZE_MODES
from .augmentation import AugmentConfig, Augmenter
from .dialog_scroll import ScrollBody
from .export_preprocess import Preprocess, ResizeConfig, resize_preview
from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT, C_GREEN, C_AMBER, C_TXT1, C_TXT2, C_TXT3,
                    C_ON_ACCENT, C_DANGER_BG, C_DANGER_FG, C_STAGE)

SPLIT_COLORS = {image_splits.TRAIN: C_GREEN, image_splits.VALID: C_AMBER, image_splits.TEST: C_ACCENT}
THUMB_W, THUMB_H = 112, 76
PREVIEW_MAX_W = 460                      # the picture the previews are made from (pixel sizes are scaled to it)
GALLERY_COLUMNS = 5


def _photo(bgr, max_w, max_h):
    """PhotoImage of the picture, shrunk to fit max_w x max_h."""
    h, w = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def _cover(bgr, w, h):
    """The picture scaled to fill w x h, centre-cropped (for the tiles of the gallery)."""
    sh, sw = bgr.shape[:2]
    s = max(w / sw, h / sh)
    big = cv2.resize(bgr, (max(w, round(sw * s)), max(h, round(sh * s))), interpolation=cv2.INTER_AREA)
    y, x = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
    return big[y:y + h, x:x + w]


def _flat(parent, text, command, kind='secondary', **kw):
    bg, fg = {'primary': (C_ACCENT, C_ON_ACCENT), 'danger': (C_DANGER_BG, C_DANGER_FG)}.get(kind, (C_CARD2, C_TXT1))
    return tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
                     relief=tk.FLAT, cursor='hand2', font=('Segoe UI', 9, 'bold'), highlightthickness=1,
                     highlightbackground=C_ACCENT if kind == 'primary' else C_BORDER, borderwidth=0, **kw)


def _check(parent, text, var, command=None, bold=False):
    return tk.Checkbutton(parent, text=text, variable=var, command=command, bg=C_CARD, fg=C_TXT1,
                          selectcolor=C_CARD2, activebackground=C_CARD, activeforeground=C_TXT1, anchor='w',
                          font=('Segoe UI', 9, 'bold' if bold else 'normal'))


# ----------------------------------------------------------------------
#  Two-handle split slider
# ----------------------------------------------------------------------
class SplitSlider(tk.Canvas):
    PAD, BAR_H, HANDLE = 12, 22, 9

    def __init__(self, parent, train=70, valid=20, width=640, on_change=None):
        super().__init__(parent, width=width, height=64, bg=C_CARD, highlightthickness=0)
        self.w, self.on_change = width, on_change
        self.train, self.valid = train, valid
        self._drag = None
        self.bind('<Button-1>', self._press)
        self.bind('<B1-Motion>', self._motion)
        self.bind('<ButtonRelease-1>', lambda e: setattr(self, '_drag', None))
        self.redraw()

    @property
    def test(self):
        return 100 - self.train - self.valid

    def set(self, train, valid):
        train = max(0, min(100, int(train)))
        self.train, self.valid = train, max(0, min(100 - train, int(valid)))
        self.redraw()

    def _x(self, pct):
        return self.PAD + (self.w - 2 * self.PAD) * pct / 100.0

    def redraw(self):
        self.delete('all')
        y1, y2 = 24, 24 + self.BAR_H
        edges = [0, self.train, self.train + self.valid, 100]
        for (a, b), split in zip(zip(edges, edges[1:]), image_splits.SPLITS):
            x1, x2 = self._x(a), self._x(b)
            if x2 - x1 >= 1:
                self.create_rectangle(x1, y1, x2, y2, fill=SPLIT_COLORS[split], outline='')
            if x2 - x1 > 70:
                self.create_text((x1 + x2) / 2, 12, text=f"{image_splits.LABELS[split]}  {b - a}%",
                                 fill=C_TXT1, font=('Segoe UI', 9, 'bold'))
        for pct in (self.train, self.train + self.valid):
            x = self._x(pct)
            self.create_oval(x - self.HANDLE, (y1 + y2) / 2 - self.HANDLE, x + self.HANDLE, (y1 + y2) / 2 + self.HANDLE,
                             fill='#FFFFFF', outline=C_TXT2, width=2)

    def _press(self, event):
        near = [(abs(event.x - self._x(p)), i) for i, p in enumerate((self.train, self.train + self.valid))]
        dist, which = min(near)
        self._drag = which if dist <= 18 else None

    def _motion(self, event):
        if self._drag is None:
            return
        pct = round((event.x - self.PAD) / (self.w - 2 * self.PAD) * 100)
        pct = max(0, min(100, pct))
        if self._drag == 0:
            edge = self.train + self.valid                  # the second handle stays where it was
            self.train = pct
            self.valid = max(0, edge - pct)
        else:
            self.valid = max(0, pct - self.train)
        self.redraw()
        if self.on_change:
            self.on_change()


# ----------------------------------------------------------------------
#  One augmentation: preview + sliders
# ----------------------------------------------------------------------
class AugmentOptionDialog:
    def __init__(self, parent, key, params, source, scale, enabled, on_apply, on_remove):
        self.key, self.op = key, augmentation.OPS[key]
        self.source, self.scale = source, scale
        self.on_apply, self.on_remove = on_apply, on_remove
        self.vars = {}
        self._after = None
        self._photos = []

        self.win = tk.Toplevel(parent)
        self.win.title(self.op.label)
        self.win.configure(bg=C_CARD)
        self.win.transient(parent)
        self.win.resizable(False, False)

        head = tk.Frame(self.win, bg=C_CARD)
        head.pack(fill=tk.X)
        tk.Label(head, text=self.op.label, bg=C_CARD, fg=C_TXT1, font=('Segoe UI', 14)).pack(side=tk.LEFT, padx=18, pady=12)
        tk.Frame(self.win, bg=C_BORDER, height=1).pack(fill=tk.X)

        body = tk.Frame(self.win, bg=C_CARD)
        body.pack(fill=tk.BOTH, expand=True)
        self.stage = tk.Frame(body, bg=C_STAGE, width=500, height=500)
        self.stage.pack(side=tk.LEFT, fill=tk.BOTH)
        self.stage.pack_propagate(False)
        side = tk.Frame(body, bg=C_PANEL, width=330, padx=16, pady=14)
        side.pack(side=tk.LEFT, fill=tk.BOTH)
        side.pack_propagate(False)

        tk.Label(side, text=self.op.description, bg=C_PANEL, fg=C_TXT2, wraplength=296, justify=tk.LEFT,
                 anchor='w', font=('Segoe UI', 10)).pack(fill=tk.X, pady=(0, 14))
        start = augmentation.clean_params(key, params)
        for p in self.op.params:
            if p.kind == 'slider':
                self._slider(side, p, start[p.name])
        for p in self.op.params:
            if p.kind == 'check':
                var = tk.BooleanVar(value=start[p.name])
                self.vars[p.name] = var
                tk.Checkbutton(side, text=p.label, variable=var, command=self.refresh, bg=C_PANEL, fg=C_TXT1,
                               selectcolor=C_CARD2, activebackground=C_PANEL, activeforeground=C_TXT1, anchor='w',
                               font=('Segoe UI', 10, 'bold')).pack(fill=tk.X, pady=2)

        tk.Frame(self.win, bg=C_BORDER, height=1).pack(fill=tk.X)
        foot = tk.Frame(self.win, bg=C_CARD)
        foot.pack(fill=tk.X, padx=14, pady=10)
        _flat(foot, "Go Back", self.win.destroy).pack(side=tk.LEFT, ipadx=12, ipady=5)
        _flat(foot, "Apply", self.apply, 'primary').pack(side=tk.RIGHT, ipadx=18, ipady=5)
        if enabled:
            _flat(foot, "Remove", self.remove, 'danger').pack(side=tk.RIGHT, padx=8, ipadx=10, ipady=5)

        self.refresh()
        self.win.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.win.winfo_reqwidth()) // 2)
        y = max(0, parent.winfo_rooty() + 30)
        self.win.geometry(f"+{x}+{y}")
        self.win.grab_set()

    def _slider(self, parent, p, value):
        row = tk.Frame(parent, bg=C_PANEL)
        row.pack(fill=tk.X, pady=(0, 10))
        top = tk.Frame(row, bg=C_PANEL)
        top.pack(fill=tk.X)
        tk.Label(top, text=p.label, bg=C_PANEL, fg=C_TXT2, font=('Segoe UI', 9), anchor='w').pack(side=tk.LEFT)
        badge = tk.Label(top, bg=C_ACCENT, fg=C_ON_ACCENT, font=('Segoe UI', 9, 'bold'), padx=6)
        badge.pack(side=tk.RIGHT)
        var = tk.DoubleVar(value=value)
        self.vars[p.name] = var

        def moved(*_):
            badge.config(text=f"{var.get():g}{p.unit}")
            self.refresh()
        tk.Scale(row, variable=var, from_=p.lo, to=p.hi, orient=tk.HORIZONTAL, resolution=1, showvalue=False,
                 bg=C_PANEL, troughcolor=C_BORDER, highlightthickness=0, bd=0, sliderlength=18,
                 activebackground=C_ACCENT).pack(fill=tk.X)
        var.trace_add('write', moved)
        badge.config(text=f"{value:g}{p.unit}")

    def params(self):
        return augmentation.clean_params(self.key, {n: v.get() for n, v in self.vars.items()})

    def refresh(self):
        if self._after:
            self.win.after_cancel(self._after)
        self._after = self.win.after(80, self._draw)

    def _draw(self):
        self._after = None
        shots = augmentation.preview(self.key, self.params(), self.source, self.scale)
        for child in self.stage.winfo_children():
            child.destroy()
        self._photos = []
        shown = [(c, i) for c, i in shots]
        original, variants = shown[0], shown[1:]
        top = tk.Frame(self.stage, bg=C_STAGE)
        top.pack(pady=(14, 6))
        self._tile(top, *original, 300, 190)
        row = tk.Frame(self.stage, bg=C_STAGE)
        row.pack(pady=4)
        width = 236 if len(variants) > 1 else 300
        for caption, img in variants:
            self._tile(row, caption, img, width, 160, side=tk.LEFT, padx=8)

    def _tile(self, parent, caption, img, max_w, max_h, **pack):
        cell = tk.Frame(parent, bg=C_STAGE)
        cell.pack(**pack)
        photo = _photo(img, max_w, max_h)
        self._photos.append(photo)
        tk.Label(cell, image=photo, bg=C_STAGE, bd=0).pack()
        tk.Label(cell, text=caption, bg=C_STAGE, fg=C_TXT2, font=('Segoe UI', 9)).pack(pady=(3, 0))

    def apply(self):
        self.on_apply(self.key, self.params())
        self.win.destroy()

    def remove(self):
        self.on_remove(self.key)
        self.win.destroy()


# ----------------------------------------------------------------------
#  The export window
# ----------------------------------------------------------------------
class ExportDialog:
    def __init__(self, gui):
        self.gui, self.root = gui, gui.root
        self.records = dataset_export.find_images(_config.datasets_root, _config.workspaceName)
        self.store = gui.split_store
        self.classes = list(_config.class_manager.get_classes())
        self.aug_ops = {}                       # key -> parameters of the enabled augmentations
        self.busy = False
        self.lockable = []
        self._photos = {}
        self.source_raw = None                  # the random picture the previews are made of
        self._tiles = {}
        self._avail = {}                        # format -> exportable records

        self.win = tk.Toplevel(self.root)
        self.win.title("Export Dataset")
        self.win.configure(bg=C_BASE)
        self.win.transient(self.root)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.win.update_idletasks()
        self.scroll.fit(self.win, 760, self.root)
        self.win.grab_set()
        self.reroll_preview()
        self.update_summary()

    # ------------------------------------------------------------ layout
    def _section(self, number, title, subtitle=""):
        outer = tk.Frame(self.scroll.inner, bg=C_CARD, highlightthickness=1, highlightbackground=C_BORDER)
        outer.pack(fill=tk.X, padx=14, pady=(0, 12))
        head = tk.Frame(outer, bg=C_CARD)
        head.pack(fill=tk.X, padx=14, pady=(12, 0))
        tk.Label(head, text=str(number), bg=C_ACCENT, fg=C_ON_ACCENT, font=('Segoe UI', 9, 'bold'), width=2).pack(side=tk.LEFT)
        tk.Label(head, text=title, bg=C_CARD, fg=C_TXT1, font=('Segoe UI', 12, 'bold')).pack(side=tk.LEFT, padx=8)
        if subtitle:
            tk.Label(outer, text=subtitle, bg=C_CARD, fg=C_TXT3, anchor='w', justify=tk.LEFT, wraplength=690,
                     font=('Segoe UI', 9)).pack(fill=tk.X, padx=14, pady=(2, 0))
        body = tk.Frame(outer, bg=C_CARD)
        body.pack(fill=tk.X, padx=14, pady=(8, 12))
        return body

    def _build(self):
        tk.Label(self.win, text="Export Dataset", bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 14, 'bold')
                 ).pack(anchor='w', padx=16, pady=(12, 8))
        self.scroll = ScrollBody(self.win, C_BASE)       # packed after the footer, so the footer always stays visible

        # --- format
        top = tk.Frame(self.scroll.inner, bg=C_BASE)
        top.pack(fill=tk.X, padx=14, pady=(0, 10))
        tk.Label(top, text="Format", bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT)
        self.fmt_var = tk.StringVar(value="YOLO")
        self.fmt_combo = ttk.Combobox(top, textvariable=self.fmt_var, values=dataset_export.FORMATS, state='readonly',
                                      width=20, font=('Segoe UI', 10), style='Jelibox.TCombobox')
        self.fmt_combo.pack(side=tk.LEFT, padx=8)
        self.fmt_combo.bind('<<ComboboxSelected>>', lambda e: self.update_summary())
        self.source_lbl = tk.Label(top, text="", bg=C_BASE, fg=C_TXT3, font=('Segoe UI', 9))
        self.source_lbl.pack(side=tk.LEFT, padx=8)
        self.lockable.append(self.fmt_combo)

        self._build_split()
        self._build_preprocess()
        self._build_augmentation()

        # --- footer
        foot = tk.Frame(self.win, bg=C_BASE)
        foot.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=10)
        tk.Frame(self.win, bg=C_BORDER, height=1).pack(side=tk.BOTTOM, fill=tk.X)
        self.summary_lbl = tk.Label(foot, text="", bg=C_BASE, fg=C_TXT1, anchor='w', justify=tk.LEFT,
                                    font=('Segoe UI', 9, 'bold'), wraplength=720)
        self.summary_lbl.pack(fill=tk.X)
        self.status_lbl = tk.Label(foot, text="", bg=C_BASE, fg=C_TXT2, anchor='w', font=('Segoe UI', 9))
        self.status_lbl.pack(fill=tk.X, pady=(4, 0))
        self.progress = ttk.Progressbar(foot, orient='horizontal', mode='determinate', maximum=100)
        self.progress.pack(fill=tk.X, pady=(4, 8))
        row = tk.Frame(foot, bg=C_BASE)
        row.pack(fill=tk.X)
        self.export_btn = _flat(row, "Export", self.start_export, 'primary')
        self.export_btn.pack(side=tk.RIGHT, ipadx=20, ipady=5)
        self.close_btn = _flat(row, "Close", self.close)
        self.close_btn.pack(side=tk.RIGHT, padx=8, ipadx=12, ipady=5)
        self.scroll.pack(fill=tk.BOTH, expand=True)

    # ---- 1. split
    def _build_split(self):
        body = self._section(1, "Train / Valid / Test split",
                             "The splits you set in Jelibox, or a new random cut. Augmentation is only added to Train.")
        self.mode_var = tk.StringVar(value='assigned')
        for value, text in (('assigned', "Use the splits assigned in Jelibox (the Train / Valid / Test buttons)"),
                            ('random', "Split randomly by percentage")):
            rb = tk.Radiobutton(body, text=text, value=value, variable=self.mode_var, command=self._on_mode,
                                bg=C_CARD, fg=C_TXT1, selectcolor=C_CARD2, activebackground=C_CARD,
                                activeforeground=C_TXT1, anchor='w', font=('Segoe UI', 10))
            rb.pack(fill=tk.X)
            self.lockable.append(rb)
        self.random_box = tk.Frame(body, bg=C_CARD)
        self.slider = SplitSlider(self.random_box, 70, 20, 640, self._on_slider)
        self.slider.pack(pady=(6, 2))
        spin = tk.Frame(self.random_box, bg=C_CARD)
        spin.pack(fill=tk.X)
        self.train_var, self.valid_var = tk.StringVar(value='70'), tk.StringVar(value='20')
        for text, var in (("Train %", self.train_var), ("Valid %", self.valid_var)):
            tk.Label(spin, text=text, bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 4))
            sb = tk.Spinbox(spin, from_=0, to=100, textvariable=var, width=5, command=self._on_spin)
            sb.pack(side=tk.LEFT, padx=(0, 14))
            sb.bind('<KeyRelease>', lambda e: self._on_spin())
            self.lockable.append(sb)
        self.test_lbl = tk.Label(spin, text="Test 10%", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9))
        self.test_lbl.pack(side=tk.LEFT)
        self.chips = tk.Frame(body, bg=C_CARD)
        self.chips.pack(fill=tk.X, pady=(8, 0))
        self.chip_lbls = {}
        for split in image_splits.SPLITS:
            chip = tk.Frame(self.chips, bg=C_CARD2, highlightthickness=1, highlightbackground=C_BORDER)
            chip.pack(side=tk.LEFT, padx=(0, 8), ipadx=8, ipady=3)
            tk.Label(chip, bg=SPLIT_COLORS[split], width=1).pack(side=tk.LEFT, padx=(0, 6))
            lbl = tk.Label(chip, text="", bg=C_CARD2, fg=C_TXT1, font=('Segoe UI', 10, 'bold'))
            lbl.pack(side=tk.LEFT)
            self.chip_lbls[split] = lbl

    def _on_mode(self):
        if self.mode_var.get() == 'random':
            self.random_box.pack(fill=tk.X, before=self.chips)
        else:
            self.random_box.pack_forget()
        self.update_summary()

    def _on_slider(self):
        self.train_var.set(str(self.slider.train))
        self.valid_var.set(str(self.slider.valid))
        self.update_summary()

    def _on_spin(self):
        digits = lambda v: int(''.join(c for c in v.get() if c.isdigit()) or 0)
        self.slider.set(digits(self.train_var), digits(self.valid_var))
        self.update_summary()

    # ---- 2. preprocessing
    def _build_preprocess(self):
        body = self._section(2, "Preprocessing", "Applied to every split.")
        self.orient_var = tk.BooleanVar(value=True)
        cb = _check(body, "Auto-Orient: discard EXIF rotation", self.orient_var, bold=True)
        cb.pack(fill=tk.X)
        self.lockable.append(cb)
        tk.Label(body, text="Pictures that carry an EXIF orientation are saved upright with the tag removed, so the "
                            "labels still fit in tools that ignore EXIF. Other pictures are copied as they are.",
                 bg=C_CARD, fg=C_TXT3, anchor='w', justify=tk.LEFT, wraplength=680, font=('Segoe UI', 9)
                 ).pack(fill=tk.X, padx=(22, 0), pady=(0, 8))
        self.resize_var = tk.BooleanVar(value=False)
        cb = _check(body, "Resize", self.resize_var, self._on_resize_toggle, bold=True)
        cb.pack(fill=tk.X)
        self.lockable.append(cb)
        row = tk.Frame(body, bg=C_CARD)
        row.pack(fill=tk.X, padx=(22, 0), pady=(2, 0))
        self.mode_names = list(RESIZE_MODES.values())
        self.resize_mode = tk.StringVar(value=self.mode_names[0])
        self.resize_combo = ttk.Combobox(row, textvariable=self.resize_mode, values=self.mode_names, state='disabled',
                                         width=28, font=('Segoe UI', 10), style='Jelibox.TCombobox')
        self.resize_combo.pack(side=tk.LEFT)
        self.resize_combo.bind('<<ComboboxSelected>>', lambda e: self._preprocess_changed())
        self.width_var, self.height_var = tk.StringVar(value='640'), tk.StringVar(value='640')
        self.size_entries = []
        for i, var in enumerate((self.width_var, self.height_var)):
            if i:
                tk.Label(row, text="×", bg=C_CARD, fg=C_TXT2).pack(side=tk.LEFT, padx=4)
            e = tk.Entry(row, textvariable=var, width=6, state='disabled', justify='center')
            e.pack(side=tk.LEFT, padx=(8 if not i else 0, 0))
            e.bind('<FocusOut>', lambda ev: self._preprocess_changed())
            e.bind('<Return>', lambda ev: self._preprocess_changed())
            self.size_entries.append(e)
        tk.Label(row, text="px", bg=C_CARD, fg=C_TXT3).pack(side=tk.LEFT, padx=4)
        self.lockable.append(self.resize_combo)
        self.lockable.extend(self.size_entries)

    def _on_resize_toggle(self):
        on = self.resize_var.get()
        self.resize_combo.config(state='readonly' if on else 'disabled')
        for e in self.size_entries:
            e.config(state='normal' if on else 'disabled')
        self._preprocess_changed()

    def _preprocess_changed(self):
        self.reroll_preview(keep=True)

    def preprocess(self):
        """Preprocess from the window; raises ValueError for a bad size."""
        resize = None
        if self.resize_var.get():
            try:
                w, h = int(self.width_var.get()), int(self.height_var.get())
            except ValueError:
                raise ValueError("The resize width and height must be whole numbers.")
            if not (8 <= w <= 8192 and 8 <= h <= 8192):
                raise ValueError("The resize width and height must be between 8 and 8192 pixels.")
            mode = next(k for k, v in RESIZE_MODES.items() if v == self.resize_mode.get())
            resize = ResizeConfig(mode, w, h)
        return Preprocess(self.orient_var.get(), resize)

    # ---- 3. augmentation
    def _build_augmentation(self):
        body = self._section(3, "Augmentation",
                             "Augmentations create new training examples for your model to learn from. Click one to "
                             "set it up; the preview uses a random picture of your dataset.")
        bar = tk.Frame(body, bg=C_CARD)
        bar.pack(fill=tk.X, pady=(0, 6))
        self.new_preview_btn = _flat(bar, "⟳  New preview picture", lambda: self.reroll_preview())
        self.new_preview_btn.pack(side=tk.LEFT, ipadx=8, ipady=2)
        self.lockable.append(self.new_preview_btn)
        grid = tk.Frame(body, bg=C_CARD)
        grid.pack(fill=tk.X)
        keys = [k for g in ("geometry", "color", "effects") for k, op in augmentation.OPS.items() if op.group == g]
        for i, key in enumerate(keys):
            cell = tk.Frame(grid, bg=C_CARD, padx=5, pady=4)
            cell.grid(row=i // GALLERY_COLUMNS, column=i % GALLERY_COLUMNS, sticky='n')
            img = tk.Label(cell, bg=C_STAGE, width=THUMB_W, height=THUMB_H, bd=0, highlightthickness=3,
                           highlightbackground=C_CARD, cursor='hand2')
            img.pack()
            name = tk.Label(cell, text=augmentation.OPS[key].label, bg=C_CARD, fg=C_TXT1, font=('Segoe UI', 9), cursor='hand2')
            name.pack(pady=(3, 0))
            for w in (img, name):
                w.bind('<Button-1>', lambda e, k=key: self.open_op(k))
            self._tiles[key] = (img, name)
        opts = tk.Frame(body, bg=C_CARD)
        opts.pack(fill=tk.X, pady=(10, 0))
        tk.Label(opts, text="Outputs per training image", bg=C_CARD, fg=C_TXT1, font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self.copies_var = tk.StringVar(value='3')
        self.copies_spin = tk.Spinbox(opts, from_=1, to=augmentation.MAX_COPIES, textvariable=self.copies_var, width=4,
                                      command=self.update_summary)
        self.copies_spin.pack(side=tk.LEFT, padx=(8, 2))
        tk.Label(opts, text="x", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self.copies_spin.bind('<KeyRelease>', lambda e: self.update_summary())
        self.keep_var = tk.BooleanVar(value=False)
        keep = _check(opts, "Keep the original as one of them", self.keep_var, self.update_summary)
        keep.pack(side=tk.LEFT, padx=(14, 0))
        seed_row = tk.Frame(body, bg=C_CARD)
        seed_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(seed_row, text="Seed (optional)", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.seed_var = tk.StringVar()
        seed = tk.Entry(seed_row, textvariable=self.seed_var, width=9)
        seed.pack(side=tk.LEFT)
        self.lockable.extend([self.copies_spin, seed, keep])
        self.applied_lbl = tk.Label(body, text="", bg=C_CARD, fg=C_TXT2, anchor='w', justify=tk.LEFT, wraplength=690,
                                    font=('Segoe UI', 9))
        self.applied_lbl.pack(fill=tk.X, pady=(8, 0))

    def reroll_preview(self, keep=False):
        """Pick a random picture of the dataset (or keep the current one) and redraw the gallery."""
        self.win.config(cursor='watch')
        self.win.update_idletasks()
        try:
            if not keep or self.source_raw is None:
                self.source_raw = None
                pool = list(self.records)
                random.shuffle(pool)
                for rec in pool[:10]:
                    self.source_raw = augmentation.read_image(rec['source'])
                    if self.source_raw is not None:
                        break
            try:
                pre = self.preprocess()
            except ValueError:
                pre = Preprocess(self.orient_var.get(), None)
            if self.source_raw is None:
                return
            prepared = resize_preview(self.source_raw, pre)
            h, w = prepared.shape[:2]
            self.scale = min(1.0, PREVIEW_MAX_W / w)
            self.preview = cv2.resize(prepared, (max(1, round(w * self.scale)), max(1, round(h * self.scale))),
                                      interpolation=cv2.INTER_AREA) if self.scale < 1.0 else prepared
            self._draw_gallery()
        finally:
            self.win.config(cursor='')

    def _draw_gallery(self):
        for key, (img, name) in self._tiles.items():
            shots = augmentation.preview(key, augmentation.default_params(key), self.preview, self.scale)
            tile = _cover(shots[-1][1], THUMB_W, THUMB_H)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)))
            self._photos[key] = photo
            img.config(image=photo, width=THUMB_W, height=THUMB_H)
        self._mark_enabled()

    def _mark_enabled(self):
        for key, (img, name) in self._tiles.items():
            on = key in self.aug_ops
            img.config(highlightbackground=C_ACCENT if on else C_CARD)
            name.config(text=("✓ " if on else "") + augmentation.OPS[key].label, fg=C_ACCENT if on else C_TXT1,
                        font=('Segoe UI', 9, 'bold' if on else 'normal'))
        lines = [augmentation.summary(k, p) for k, p in self.aug_ops.items()]
        self.applied_lbl.config(text=("Applied: " + "  ·  ".join(lines)) if lines else "No augmentation selected.")
        self.update_summary()

    def open_op(self, key):
        if self.busy or self.source_raw is None:
            return
        AugmentOptionDialog(self.win, key, self.aug_ops.get(key) or augmentation.default_params(key), self.preview,
                            self.scale, key in self.aug_ops, self._apply_op, self._remove_op)

    def _apply_op(self, key, params):
        self.aug_ops[key] = params
        self._mark_enabled()

    def _remove_op(self, key):
        self.aug_ops.pop(key, None)
        self._mark_enabled()

    # ------------------------------------------------------------ numbers
    def _available(self, fmt):
        if fmt not in self._avail:
            self._avail[fmt] = dataset_export.exportable(fmt, self.records, _config.yolo_labels_folder)
        return self._avail[fmt]

    def counts(self):
        records = self._available(self.fmt_var.get())
        if self.mode_var.get() == 'assigned':
            return dataset_export.assigned_counts(records, self.store)
        return dataset_export.random_counts(len(records), self.slider.train, self.slider.valid)

    def copies(self):
        try:
            return max(1, min(augmentation.MAX_COPIES, int(self.copies_var.get())))
        except ValueError:
            return 1

    def update_summary(self):
        fmt = self.fmt_var.get()
        have, total = len(self._available(fmt)), len(self.records)
        self.source_lbl.config(text=f"{have} of {total} images can be exported" if have != total else f"{total} images")
        self.test_lbl.config(text=f"Test {self.slider.test}%")
        counts = self.counts()
        times = self.copies() if self.aug_ops else 1
        for split, n in zip(image_splits.SPLITS, counts):
            pct = f" · {round(100 * n / have)}%" if have else ""
            self.chip_lbls[split].config(text=f"{image_splits.LABELS[split]}  {n}{pct}")
        train = f"Train {counts[0]}" + (f" → {counts[0] * times} images ({times}x)" if self.aug_ops else "")
        parts = [train, f"Valid {counts[1]}", f"Test {counts[2]}"]
        self.summary_lbl.config(text="Export: " + "  ·  ".join(parts))
        if not self.busy:
            self.export_btn.config(state='normal' if have else 'disabled')

    # ------------------------------------------------------------ export
    def _set_busy(self, busy):
        self.busy = busy
        state = 'disabled' if busy else 'normal'
        for w in self.lockable:
            try:
                w.config(state=state if not isinstance(w, ttk.Combobox) else ('disabled' if busy else 'readonly'))
            except tk.TclError:
                pass
        if not busy:
            self._on_resize_toggle_state()
        self.export_btn.config(state=state)
        self.close_btn.config(state=state)

    def _on_resize_toggle_state(self):
        on = self.resize_var.get()
        self.resize_combo.config(state='readonly' if on else 'disabled')
        for e in self.size_entries:
            e.config(state='normal' if on else 'disabled')

    def close(self):
        if not self.busy:
            self.win.destroy()

    def start_export(self):
        if self.busy:
            return
        try:
            pre = self.preprocess()
            seed_text = self.seed_var.get().strip()
            if seed_text and not seed_text.lstrip('-').isdigit():
                raise ValueError("Seed must be a whole number (or empty).")
        except ValueError as exc:
            messagebox.showerror("Export Dataset", str(exc), parent=self.win)
            return
        seed = int(seed_text) if seed_text else None
        fmt = self.fmt_var.get()
        records = self._available(fmt)
        by_split = dataset_export.split_records(records, self.mode_var.get(), self.store, self.slider.train,
                                                self.slider.valid, seed)
        augmenter = None
        if self.aug_ops:
            keep = self.keep_var.get()
            augmenter = Augmenter(AugmentConfig(ops=dict(self.aug_ops), copies=self.copies() - (1 if keep else 0),
                                                keep_original=keep, seed=seed, resize=pre.resize))
        classes = list(self.classes)
        target = _config.export_dataset_folder
        voc, labels = _config.vocdataset_folder, _config.yolo_labels_folder

        self._set_busy(True)
        self.progress['value'] = 0
        self.status_lbl.config(text="Preparing export...")
        self.win.update_idletasks()

        box = {'progress': (0, 1, None), 'result': None, 'error': None}      # plain data: the worker never touches Tk

        def progress_cb(current, total, message=None):
            box['progress'] = (current, total, message)

        def worker():
            try:
                out = target
                try:
                    if os.path.exists(out):
                        project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                        if os.path.abspath(out).startswith(project):
                            shutil.rmtree(os.path.abspath(out))     # never mix with a previous export
                        else:
                            out = os.path.join(out, f'export_{int(time.time())}')
                except OSError:
                    pass
                os.makedirs(out, exist_ok=True)
                result = dataset_export.export_dataset(fmt, out, by_split, classes, voc, labels, pre, augmenter,
                                                       progress_cb)
                c = result['counts']
                text = (f"Dataset exported to:\n{out}\n\n"
                        f"Train {c['train']}  ·  Valid {c['valid']}  ·  Test {c['test']}")
                if augmenter is not None:
                    text += f"\nAugmented outputs written to Train: {result['augmented']}"
                    if result['skipped']:
                        text += f" ({result['skipped']} skipped: nothing changed)"
                box['result'] = (out, text)
            except Exception as exc:
                box['error'] = str(exc)

        def poll():
            current, total, message = box['progress']
            pct = int(current / total * 100) if total else 0
            self.progress['value'] = pct
            self.status_lbl.config(text=message or f"Exporting... {pct}%")
            if box['result']:
                out, text = box['result']
                self._set_busy(False)
                self.progress['value'] = 100
                self.status_lbl.config(text=f"Export completed to: {out}")
                messagebox.showinfo("Export Completed", text, parent=self.win)
            elif box['error'] is not None:
                self._set_busy(False)
                self.progress['value'] = 0
                self.status_lbl.config(text="Export failed")
                messagebox.showerror("Export Failed", f"Error during export:\n{box['error']}", parent=self.win)
            else:
                self.win.after(100, poll)

        threading.Thread(target=worker, daemon=True).start()
        self.win.after(100, poll)


def open_export_dialog(gui):
    ExportDialog(gui)
