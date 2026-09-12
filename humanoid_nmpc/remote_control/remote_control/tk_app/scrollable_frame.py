"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """
    A reusable frame that embeds a scrollable canvas with a vertical scrollbar,
    full cross-platform mouse wheel support (Linux, Windows, macOS), and dynamic resizing.
    """

    def __init__(self, parent, bg_color="#2c2c2c", *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.configure(style="TFrame")

        # Canvas with matching dark background
        self.canvas = tk.Canvas(
            self,
            bg=bg_color,
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_content = ttk.Frame(self.canvas)

        self.scrollable_content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_content, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Stretch canvas to fill width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel for scrolling (Linux, macOS, Windows)
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.scrollable_content)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, widget):
        widget.bind("<Enter>", lambda _: self._activate_mousewheel(widget))
        widget.bind("<Leave>", lambda _: self._deactivate_mousewheel(widget))

    def _activate_mousewheel(self, widget):
        # Linux (X11) mouse wheel buttons
        widget.bind_all("<Button-4>", self._on_mousewheel_up)
        widget.bind_all("<Button-5>", self._on_mousewheel_down)
        # Windows / macOS mouse wheel
        widget.bind_all("<MouseWheel>", self._on_mousewheel)

    def _deactivate_mousewheel(self, widget):
        widget.unbind_all("<Button-4>")
        widget.unbind_all("<Button-5>")
        widget.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        delta = -1 if event.delta < 0 else 1
        self.canvas.yview_scroll(-delta * 2, "units")

    def _on_mousewheel_up(self, event):
        self.canvas.yview_scroll(-2, "units")

    def _on_mousewheel_down(self, event):
        self.canvas.yview_scroll(2, "units")
