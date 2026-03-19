"""
GUI Node — ARIA Emotion Display Robot
Huge animated face dominates screen, single message input at bottom
"""

import tkinter as tk
from tkinter import Canvas
import threading
import time
import math
import queue


class GUINode:
    def __init__(self, state_queue, emotion_queue, command_queue):
        self.state_queue   = state_queue
        self.emotion_queue = emotion_queue
        self.command_queue = command_queue

        self.root = tk.Tk()
        self.root.title("ARIA — Reception Assistant")
        self.root.geometry("1280x800+0+0")
        self.root.configure(bg='#0A0E1A')
        self.root.resizable(True, True)

        self.current_emotion = 'NEUTRAL'
        self.running         = True
        self.blink_open      = True
        self.anim_t          = 0.0
        self.anim_y          = 0
        self._msg_count      = 0

        # Dark professional palette
        self.c = {
            'bg':        '#0A0E1A',
            'surface':   '#141929',
            'surface2':  '#1C2333',
            'border':    '#2A3450',
            'ink':       '#E8EDF5',
            'ink2':      '#A8B4CC',
            'ink3':      '#5A6880',
            'blue':      '#3B82F6',
            'blue2':     '#2563EB',
            'blue_glow': '#1E3A5F',
            'green':     '#10B981',
            'green_bg':  '#064E3B',
            'face_bg':   '#0D1220',
            # emotion colours
            'NEUTRAL':   '#3B82F6',
            'HAPPY':     '#10B981',
            'SAD':       '#6366F1',
            'ANGRY':     '#EF4444',
            'SURPRISED': '#A855F7',
            'CONFUSED':  '#F59E0B',
            'TIRED':     '#6B7280',
            'EXCITED':   '#06B6D4',
            'ATTENTIVE': '#3B82F6',
        }

        self._build()
        threading.Thread(target=self._poll,       daemon=True).start()
        threading.Thread(target=self._blink_loop, daemon=True).start()
        threading.Thread(target=self._float_loop, daemon=True).start()
        self._redraw('NEUTRAL')
        self._add_msg('ROBOT', 'Good day! I am ARIA, your reception assistant. How may I help you?')

    # ═══════════════════════════════════════════════════════
    #  BUILD
    # ═══════════════════════════════════════════════════════

    def _build(self):
        r = self.root

        # ── TOP BAR ─────────────────────────────────────────
        bar = tk.Frame(r, bg=self.c['surface'], height=52)
        bar.pack(side='top', fill='x')
        bar.pack_propagate(False)

        lf = tk.Frame(bar, bg=self.c['surface'])
        lf.pack(side='left', padx=20, pady=10)

        logo = tk.Frame(lf, bg=self.c['blue'], width=32, height=32)
        logo.pack(side='left', padx=(0, 10))
        logo.pack_propagate(False)
        tk.Label(logo, text='A', font=('Georgia', 14, 'bold'),
                 bg=self.c['blue'], fg='white').pack(expand=True)

        tk.Label(lf, text='ARIA', font=('Georgia', 16, 'bold'),
                 bg=self.c['surface'], fg=self.c['ink']).pack(side='left')
        tk.Frame(lf, bg=self.c['border'], width=1, height=18).pack(side='left', padx=10)
        tk.Label(lf, text='Emotion Display Robot',
                 font=('Helvetica', 9), bg=self.c['surface'],
                 fg=self.c['ink3']).pack(side='left')

        rf = tk.Frame(bar, bg=self.c['surface'])
        rf.pack(side='right', padx=20, pady=10)

        self.clock_lbl = tk.Label(rf, font=('Helvetica', 10),
                                  bg=self.c['surface'], fg=self.c['ink3'])
        self.clock_lbl.pack(side='right', padx=(10, 0))
        self._tick()

        dot = tk.Frame(rf, bg=self.c['surface'])
        dot.pack(side='right')
        tk.Label(dot, text='● ONLINE', font=('Helvetica', 8, 'bold'),
                 bg=self.c['surface'], fg=self.c['green']).pack()

        tk.Frame(r, bg=self.c['border'], height=1).pack(side='top', fill='x')

        # ── MAIN BODY ────────────────────────────────────────
        body = tk.Frame(r, bg=self.c['bg'])
        body.pack(side='top', fill='both', expand=True)

        # Left: HUGE face column
        left = tk.Frame(body, bg=self.c['bg'], width=560)
        left.pack(side='left', fill='y')
        left.pack_propagate(False)
        self._build_face(left)

        # Divider
        tk.Frame(body, bg=self.c['border'], width=1).pack(side='left', fill='y')

        # Right: chat + input
        right = tk.Frame(body, bg=self.c['bg'])
        right.pack(side='left', fill='both', expand=True)
        self._build_right(right)

    def _build_face(self, p):
        # Emotion badge at top
        top = tk.Frame(p, bg=self.c['bg'])
        top.pack(fill='x', padx=20, pady=(16, 0))

        self.emotion_pill = tk.Label(
            top, text='● STANDBY',
            font=('Helvetica', 9, 'bold'),
            bg=self.c['blue_glow'], fg=self.c['blue'],
            padx=14, pady=5)
        self.emotion_pill.pack(side='left')

        self.msg_count_lbl = tk.Label(
            top, text='',
            font=('Helvetica', 8),
            bg=self.c['bg'], fg=self.c['ink3'])
        self.msg_count_lbl.pack(side='right')

        # HUGE face canvas — takes up most of left column
        face_frame = tk.Frame(p, bg=self.c['face_bg'],
                              highlightbackground=self.c['border'],
                              highlightthickness=1)
        face_frame.pack(fill='both', expand=True, padx=16, pady=12)

        self.canvas = Canvas(face_frame,
                             bg=self.c['face_bg'],
                             highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # Status text below face
        self.face_status = tk.Label(
            p, text='Awaiting visitor',
            font=('Helvetica', 11),
            bg=self.c['bg'], fg=self.c['ink3'])
        self.face_status.pack(pady=(0, 8))

        # Visitor info strip at bottom of left column
        vi = tk.Frame(p, bg=self.c['surface2'],
                      highlightbackground=self.c['border'],
                      highlightthickness=1)
        vi.pack(fill='x', padx=16, pady=(0, 16))

        row = tk.Frame(vi, bg=self.c['surface2'])
        row.pack(fill='x', padx=16, pady=10)

        for col_data in [('👤', 'name', '—'), ('🔁', 'visits', '—'), ('💬', 'mood', '—')]:
            icon, attr, val = col_data
            cell = tk.Frame(row, bg=self.c['surface2'])
            cell.pack(side='left', expand=True)
            tk.Label(cell, text=icon, font=('Helvetica', 14),
                     bg=self.c['surface2']).pack()
            lbl = tk.Label(cell, text=val,
                           font=('Helvetica', 10, 'bold'),
                           bg=self.c['surface2'], fg=self.c['ink'])
            lbl.pack()
            setattr(self, f'lbl_{attr}', lbl)

    def _build_right(self, p):
        # ── Single message input at bottom ───────────────────
        inp = tk.Frame(p, bg=self.c['surface'],
                       highlightbackground=self.c['border'],
                       highlightthickness=1)
        inp.pack(side='bottom', fill='x', padx=16, pady=16)

        tk.Label(inp, text='SEND A MESSAGE',
                 font=('Helvetica', 8, 'bold'),
                 bg=self.c['surface'], fg=self.c['ink3']).pack(
                     anchor='w', padx=14, pady=(12, 2))

        tk.Label(inp, text='Type or speak — powered by Mistral AI',
                 font=('Helvetica', 7),
                 bg=self.c['surface'], fg=self.c['ink3']).pack(
                     anchor='w', padx=14)

        entry_row = tk.Frame(inp, bg=self.c['surface'])
        entry_row.pack(fill='x', padx=14, pady=(8, 14))

        ef = tk.Frame(entry_row, bg=self.c['border'], padx=1, pady=1)
        ef.pack(side='left', fill='x', expand=True, padx=(0, 10))

        self.msg_entry = tk.Entry(
            ef,
            bg=self.c['surface2'], fg=self.c['ink'],
            font=('Helvetica', 13), relief='flat',
            insertbackground=self.c['blue'],
            disabledbackground=self.c['surface2'])
        self.msg_entry.pack(fill='x', ipady=11, ipadx=10)

        ph = 'Ask me anything…'
        self._ph_set(self.msg_entry, ph)
        self.msg_entry.bind('<FocusIn>',  lambda e: self._ph_in(self.msg_entry, ph))
        self.msg_entry.bind('<FocusOut>', lambda e: self._ph_out(self.msg_entry, ph))
        self.msg_entry.bind('<Return>', self._submit_msg)

        tk.Button(entry_row, text='Send →',
                  font=('Helvetica', 11, 'bold'),
                  bg=self.c['blue'], fg='white',
                  relief='flat', padx=20, pady=10,
                  cursor='hand2', bd=0,
                  command=self._submit_msg,
                  activebackground=self.c['blue2'],
                  activeforeground='white').pack(side='left')

        # ── Chat above input ─────────────────────────────────
        cc = tk.Frame(p, bg=self.c['bg'])
        cc.pack(side='top', fill='both', expand=True, padx=16, pady=(16, 0))

        hdr = tk.Frame(cc, bg=self.c['bg'])
        hdr.pack(fill='x', pady=(0, 8))
        tk.Label(hdr, text='Conversation',
                 font=('Georgia', 13, 'bold'),
                 bg=self.c['bg'], fg=self.c['ink']).pack(side='left')

        tk.Frame(cc, bg=self.c['border'], height=1).pack(fill='x', pady=(0, 4))

        scr = tk.Scrollbar(cc, relief='flat', width=3,
                           bg=self.c['bg'], troughcolor=self.c['bg'])
        scr.pack(side='right', fill='y')

        self.chat_box = tk.Text(
            cc,
            bg=self.c['bg'], fg=self.c['ink2'],
            font=('Helvetica', 11), wrap='word',
            state='disabled', relief='flat',
            padx=4, pady=4,
            yscrollcommand=scr.set,
            cursor='arrow', highlightthickness=0,
            spacing1=2, spacing3=8)
        self.chat_box.pack(side='left', fill='both', expand=True)
        scr.config(command=self.chat_box.yview)

        self.chat_box.tag_config('r_lbl',
                                 foreground=self.c['blue'],
                                 font=('Helvetica', 8, 'bold'))
        self.chat_box.tag_config('u_lbl',
                                 foreground=self.c['ink3'],
                                 font=('Helvetica', 8, 'bold'))
        self.chat_box.tag_config('r_msg',
                                 foreground=self.c['ink'],
                                 font=('Helvetica', 11),
                                 lmargin1=8, lmargin2=8)
        self.chat_box.tag_config('u_msg',
                                 foreground=self.c['ink2'],
                                 font=('Helvetica', 11),
                                 lmargin1=8, lmargin2=8)

    # ═══════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════

    def _tick(self):
        self.clock_lbl.config(text=time.strftime('%H:%M'))
        self.root.after(30000, self._tick)

    def _ph_set(self, e, ph):
        e.insert(0, ph)
        e.config(fg=self.c['ink3'])

    def _ph_in(self, e, ph):
        if e.get() == ph:
            e.delete(0, 'end')
            e.config(fg=self.c['ink'])

    def _ph_out(self, e, ph):
        if not e.get():
            e.insert(0, ph)
            e.config(fg=self.c['ink3'])

    def _submit_msg(self, _=None):
        ph = 'Ask me anything…'
        v  = self.msg_entry.get().strip()
        if not v or v == ph:
            return
        self._add_msg('USER', v)
        self.command_queue.put({'type': 'USER_MESSAGE', 'text': v})
        self.msg_entry.delete(0, 'end')

    def _add_msg(self, sender, msg):
        self.chat_box.config(state='normal')
        if sender == 'ROBOT':
            self.chat_box.insert('end', '\nARIA\n', 'r_lbl')
            self.chat_box.insert('end', f'{msg}\n', 'r_msg')
        else:
            self.chat_box.insert('end', '\nYou\n', 'u_lbl')
            self.chat_box.insert('end', f'{msg}\n', 'u_msg')
        self.chat_box.config(state='disabled')
        self.chat_box.see('end')
        self._msg_count += 1
        n = self._msg_count
        self.msg_count_lbl.config(
            text=f'{n} msg{"s" if n != 1 else ""}')

    def add_chat_message(self, sender, msg):
        self.root.after(0, lambda s=sender, m=msg: self._add_msg(s, m))

    def update_visitor_info(self, name=None, visits=None, mood=None):
        def _do():
            if name is not None:
                self.lbl_name.config(text=name)
            if visits is not None:
                self.lbl_visits.config(text=str(visits))
            if mood is not None:
                self.lbl_mood.config(text=mood.capitalize() if mood != '—' else '—')
        self.root.after(0, _do)

    # ═══════════════════════════════════════════════════════
    #  QUEUE POLLING
    # ═══════════════════════════════════════════════════════

    def _poll(self):
        while self.running:
            try:
                if not self.state_queue.empty():
                    self.state_queue.get_nowait()
                if not self.emotion_queue.empty():
                    e = self.emotion_queue.get_nowait()
                    self.current_emotion = e
                    self.root.after(0, lambda em=e: self._redraw(em))
            except Exception:
                pass
            time.sleep(0.05)

    # ═══════════════════════════════════════════════════════
    #  FACE ANIMATION
    # ═══════════════════════════════════════════════════════

    def _blink_loop(self):
        import random
        while self.running:
            time.sleep(random.uniform(2.5, 6))
            self.blink_open = False
            self.root.after(0, lambda: self._redraw(self.current_emotion))
            time.sleep(0.08)
            self.blink_open = True
            self.root.after(0, lambda: self._redraw(self.current_emotion))

    def _float_loop(self):
        while self.running:
            self.anim_t += 0.04
            self.anim_y = int(math.sin(self.anim_t) * 5)
            self.root.after(0, lambda: self._redraw(self.current_emotion))
            time.sleep(0.05)

    def _redraw(self, emotion):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        self.canvas.delete('all')
        col  = self.c.get(emotion, self.c['blue'])
        bg   = self.c['face_bg']

        # Face center — use actual canvas dimensions
        cx = w // 2
        cy = h // 2 + self.anim_y

        # Radius scales with canvas size
        R  = min(w, h) // 2 - 20

        # Outer glow
        self.canvas.create_oval(cx-R-16, cy-R-16, cx+R+16, cy+R+16,
                                outline=self._mix(col, '#000000', 0.5),
                                width=20, fill=bg)
        # Second glow
        self.canvas.create_oval(cx-R-6, cy-R-6, cx+R+6, cy+R+6,
                                outline=self._mix(col, '#000000', 0.3),
                                width=8, fill=bg)
        # Main circle
        self.canvas.create_oval(cx-R, cy-R, cx+R, cy+R,
                                outline=col, width=3, fill=bg)

        self._draw_eyes(cx, cy, col, emotion, R)
        self._draw_mouth(cx, cy, col, emotion, R)

        # Nose
        nr = max(4, R // 20)
        self.canvas.create_oval(cx-nr, cy+R//12, cx+nr, cy+R//6,
                                fill=self._mix(col, '#FFFFFF', 0.4), outline='')

        # Update pill
        info = {
            'NEUTRAL':   ('● STANDBY',   'Awaiting visitor'),
            'HAPPY':     ('● HAPPY',     'Glad to see you!'),
            'SAD':       ('● CONCERNED', 'Here to help.'),
            'ANGRY':     ('● ALERT',     'Let me resolve this.'),
            'SURPRISED': ('● SURPRISED', 'That is unexpected!'),
            'CONFUSED':  ('● THINKING',  'Processing…'),
            'TIRED':     ('● RESTING',   'Brief pause.'),
            'EXCITED':   ('● ENGAGED',   'Happy to assist!'),
            'ATTENTIVE': ('● LISTENING', 'I am listening.'),
        }
        pill_txt, status = info.get(emotion, ('● STANDBY', 'Ready'))
        glow = self._mix(col, self.c['bg'], 0.75)
        self.emotion_pill.config(text=pill_txt, bg=glow, fg=col)
        self.face_status.config(text=status)

        moods = {
            'NEUTRAL': 'Neutral',   'HAPPY': 'Happy',
            'SAD': 'Sad',           'ANGRY': 'Concerned',
            'SURPRISED': 'Surprised', 'CONFUSED': 'Thinking',
            'TIRED': 'Resting',     'EXCITED': 'Excited',
            'ATTENTIVE': 'Attentive',
        }
        self.lbl_mood.config(text=moods.get(emotion, '—'))

    def _draw_eyes(self, cx, cy, col, emotion, R):
        ey  = cy - R // 3
        off = R // 3
        lx, rx = cx - off, cx + off
        r = R // 6

        if not self.blink_open:
            for ex in (lx, rx):
                self.canvas.create_line(ex-r, ey, ex+r, ey,
                                        fill=col, width=max(3, R//30),
                                        capstyle='round')
            return

        lw = max(3, R // 28)

        if emotion in ('HAPPY', 'EXCITED'):
            for ex in (lx, rx):
                self.canvas.create_arc(ex-r, ey-r, ex+r, ey+r,
                                       start=0, extent=180,
                                       style='arc', outline=col, width=lw)

        elif emotion == 'SAD':
            for ex in (lx, rx):
                self.canvas.create_arc(ex-r, ey-r, ex+r, ey+r,
                                       start=180, extent=180,
                                       style='arc', outline=col, width=lw)
            brow = R // 18
            self.canvas.create_line(lx-r, ey-brow*2, lx+r, ey-brow,
                                    fill=col, width=lw-1)
            self.canvas.create_line(rx-r, ey-brow, rx+r, ey-brow*2,
                                    fill=col, width=lw-1)

        elif emotion == 'ANGRY':
            self.canvas.create_line(lx-r, ey-r//1.2, lx+r, ey-r//3,
                                    fill=col, width=lw)
            self.canvas.create_line(rx-r, ey-r//3, rx+r, ey-r//1.2,
                                    fill=col, width=lw)
            for ex in (lx, rx):
                self.canvas.create_rectangle(ex-r, ey-r//3, ex+r, ey+r,
                                             fill=col, outline='')

        elif emotion == 'SURPRISED':
            for ex in (lx, rx):
                self.canvas.create_oval(ex-r, ey-r, ex+r, ey+r,
                                        outline=col, width=lw, fill='')
                pr = r // 2
                self.canvas.create_oval(ex-pr, ey-pr, ex+pr, ey+pr,
                                        fill=col, outline='')

        elif emotion == 'TIRED':
            for ex in (lx, rx):
                self.canvas.create_arc(ex-r, ey-r, ex+r, ey+r,
                                       start=0, extent=180,
                                       fill=col, outline='')

        elif emotion == 'CONFUSED':
            self.canvas.create_oval(lx-r, ey-r, lx+r, ey+r,
                                    outline=col, width=lw, fill='')
            pr = r // 2
            self.canvas.create_oval(lx-pr, ey-pr, lx+pr, ey+pr,
                                    fill=col, outline='')
            self.canvas.create_arc(rx-r, ey-r-r//2, rx+r, ey+r-r//2,
                                   start=0, extent=180,
                                   style='arc', outline=col, width=lw)

        else:  # NEUTRAL / ATTENTIVE
            for ex in (lx, rx):
                self.canvas.create_oval(ex-r, ey-r, ex+r, ey+r,
                                        outline=col, width=lw, fill='')
                pr = r // 2
                self.canvas.create_oval(ex-pr, ey-pr, ex+pr, ey+pr,
                                        fill=col, outline='')
                shine = r // 4
                self.canvas.create_oval(ex-pr+shine, ey-pr,
                                        ex-pr+shine*2, ey-pr+shine,
                                        fill='white', outline='')

    def _draw_mouth(self, cx, cy, col, emotion, R):
        my = cy + R // 2
        mw = R // 2
        lw = max(3, R // 28)

        if emotion in ('HAPPY', 'EXCITED'):
            self.canvas.create_arc(cx-mw, my-mw//3, cx+mw, my+mw//2,
                                   start=0, extent=-180,
                                   style='arc', outline=col, width=lw)

        elif emotion == 'SAD':
            self.canvas.create_arc(cx-mw+mw//4, my+mw//4,
                                   cx+mw-mw//4, my+mw*3//4,
                                   start=0, extent=180,
                                   style='arc', outline=col, width=lw)

        elif emotion == 'ANGRY':
            self.canvas.create_line(cx-mw+mw//4, my+mw//4,
                                    cx+mw-mw//4, my+mw//4,
                                    fill=col, width=lw, capstyle='round')

        elif emotion == 'SURPRISED':
            or_ = mw // 3
            self.canvas.create_oval(cx-or_, my-or_//2, cx+or_, my+or_,
                                    fill='', outline=col, width=lw)

        elif emotion == 'CONFUSED':
            pts = [cx-mw+mw//4, my+mw//4,
                   cx-mw//4,    my,
                   cx,          my+mw//3,
                   cx+mw//4,    my,
                   cx+mw-mw//4, my+mw//4]
            self.canvas.create_line(pts, fill=col, width=lw,
                                    smooth=True, capstyle='round')

        elif emotion == 'TIRED':
            self.canvas.create_arc(cx-mw//2, my, cx+mw//2, my+mw//3,
                                   start=0, extent=180,
                                   style='arc', outline=col, width=lw-1)
        else:
            self.canvas.create_arc(cx-mw//2, my-mw//6, cx+mw//2, my+mw//4,
                                   start=0, extent=-180,
                                   style='arc', outline=col, width=lw-1)

    def _mix(self, c1, c2, t):
        try:
            def p(h):
                h = h.lstrip('#')
                return int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
            r1, g1, b1 = p(c1)
            r2, g2, b2 = p(c2)
            return '#{:02x}{:02x}{:02x}'.format(
                int(r1+(r2-r1)*t),
                int(g1+(g2-g1)*t),
                int(b1+(b2-b1)*t))
        except Exception:
            return c1

    def run(self):
        self.root.mainloop()

    def shutdown(self):
        self.running = False
        try:
            self.root.quit()
        except Exception:
            pass
