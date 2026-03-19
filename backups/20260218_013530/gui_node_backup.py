"""
GUI Node — ARIA Reception Assistant
Professional interface, guaranteed visible inputs
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
    self.root.geometry("1280x720")
        self.root.update()
        # Force window to top of screen
        self.root.geometry("1280x720+0+0")
        self.root.configure(bg='#F4F5F7')
        self.root.resizable(True, True)

        self.current_emotion = 'NEUTRAL'
        self.running         = True
        self.blink_open      = True
        self.anim_t          = 0.0
        self.anim_y          = 0
        self._msg_count      = 0

        self.c = {
            'bg':       '#F4F5F7',
            'surface':  '#FFFFFF',
            'border':   '#E1E4E8',
            'ink':      '#111827',
            'ink2':     '#374151',
            'ink3':     '#6B7280',
            'ink4':     '#9CA3AF',
            'blue':     '#2563EB',
            'blue2':    '#1D4ED8',
            'blue_bg':  '#EFF6FF',
            'green':    '#059669',
            'green_bg': '#ECFDF5',
            'face_bg':  '#F0F4FF',
            # emotion colours
            'NEUTRAL':   '#2563EB',
            'HAPPY':     '#059669',
            'SAD':       '#3B82F6',
            'ANGRY':     '#EF4444',
            'SURPRISED': '#8B5CF6',
            'CONFUSED':  '#F59E0B',
            'TIRED':     '#6B7280',
            'EXCITED':   '#10B981',
            'ATTENTIVE': '#2563EB',
        }

        self._build()

        threading.Thread(target=self._poll,       daemon=True).start()
        threading.Thread(target=self._blink_loop, daemon=True).start()
        threading.Thread(target=self._float_loop, daemon=True).start()

        self._redraw('NEUTRAL')
        self._add_msg('ROBOT', 'Good day! I am ARIA, your reception assistant. How may I help you?')

    # ═══════════════════════════════════════════════════════
    #  BUILD — top bar + two-column body
    # ═══════════════════════════════════════════════════════

    def _build(self):
        r = self.root

        # ── TOP BAR ─────────────────────────────────────────
        bar = tk.Frame(r, bg=self.c['surface'], height=56)
        bar.pack(side='top', fill='x')
        bar.pack_propagate(False)

        # Logo + title
        lf = tk.Frame(bar, bg=self.c['surface'])
        lf.pack(side='left', padx=24, pady=8)

        logo = tk.Frame(lf, bg=self.c['blue'], width=34, height=34)
        logo.pack(side='left', padx=(0, 10))
        logo.pack_propagate(False)
        tk.Label(logo, text='A', font=('Georgia', 15, 'bold'),
                 bg=self.c['blue'], fg='white').pack(expand=True)

        tk.Label(lf, text='ARIA',
                 font=('Georgia', 17, 'bold'),
                 bg=self.c['surface'], fg=self.c['ink']).pack(side='left')

        tk.Frame(lf, bg=self.c['border'],
                 width=1, height=20).pack(side='left', padx=12)

        tk.Label(lf, text='Reception Assistant',
                 font=('Helvetica', 10),
                 bg=self.c['surface'],
                 fg=self.c['ink3']).pack(side='left')

        # Right side of bar
        rf = tk.Frame(bar, bg=self.c['surface'])
        rf.pack(side='right', padx=24, pady=8)

        self.clock_lbl = tk.Label(rf, font=('Helvetica', 11),
                                  bg=self.c['surface'], fg=self.c['ink3'])
        self.clock_lbl.pack(side='right', padx=(12, 0))
        self._tick()

        pill = tk.Frame(rf, bg=self.c['green_bg'], padx=12, pady=4)
        pill.pack(side='right')
        tk.Label(pill, text='● Online',
                 font=('Helvetica', 9, 'bold'),
                 bg=self.c['green_bg'], fg=self.c['green']).pack()

        # Divider
        tk.Frame(r, bg=self.c['border'], height=1).pack(side='top', fill='x')

        # ── BODY ────────────────────────────────────────────
        body = tk.Frame(r, bg=self.c['bg'])
        body.pack(side='top', fill='both', expand=True, padx=20, pady=16)

        # LEFT column (face + visitor info) — fixed width
        left = tk.Frame(body, bg=self.c['bg'], width=300)
        left.pack(side='left', fill='y', padx=(0, 16))
        left.pack_propagate(False)
        self._build_left(left)

        # RIGHT column (chat + inputs) — expands
        right = tk.Frame(body, bg=self.c['bg'])
        right.pack(side='left', fill='both', expand=True)
        self._build_right(right)

    # ── LEFT COLUMN ─────────────────────────────────────────

    def _build_left(self, p):

        # Face card
        fc = self._card(p)
        fc.pack(fill='x', pady=(0, 12))

        # Emotion badge
        br = tk.Frame(fc, bg=self.c['surface'])
        br.pack(fill='x', padx=16, pady=(14, 0))
        self.emotion_pill = tk.Label(
            br, text='STANDBY',
            font=('Helvetica', 8, 'bold'),
            bg=self.c['blue_bg'], fg=self.c['blue'],
            padx=10, pady=3)
        self.emotion_pill.pack(side='left')

        # Face canvas
        ff = tk.Frame(fc, bg=self.c['face_bg'],
                      highlightbackground=self.c['border'],
                      highlightthickness=1,
                      width=200, height=200)
        ff.pack(pady=10)
        ff.pack_propagate(False)

        self.canvas = Canvas(ff, width=200, height=200,
                             bg=self.c['face_bg'],
                             highlightthickness=0)
        self.canvas.pack()

        tk.Frame(fc, bg=self.c['border'], height=1).pack(fill='x', padx=16)

        self.face_status = tk.Label(
            fc, text='Awaiting visitor',
            font=('Helvetica', 10),
            bg=self.c['surface'], fg=self.c['ink3'])
        self.face_status.pack(pady=10)

        # Visitor info card
        ic = self._card(p)
        ic.pack(fill='x')

        tk.Label(ic, text='VISITOR INFO',
                 font=('Helvetica', 8, 'bold'),
                 bg=self.c['surface'],
                 fg=self.c['ink4']).pack(anchor='w', padx=16, pady=(12, 6))

        tk.Frame(ic, bg=self.c['border'], height=1).pack(fill='x', padx=16)

        self.lbl_name   = self._info_row(ic, '👤', 'Name',         '—')
        self.lbl_visits = self._info_row(ic, '🔁', 'Visit Count',  '—')
        self.lbl_mood   = self._info_row(ic, '💬', 'Detected Mood','—')
        tk.Frame(ic, bg=self.c['surface'], height=4).pack()

    def _info_row(self, parent, icon, label, val):
        row = tk.Frame(parent, bg=self.c['surface'])
        row.pack(fill='x', padx=16, pady=5)

        ib = tk.Frame(row, bg=self.c['blue_bg'], width=30, height=30)
        ib.pack(side='left', padx=(0, 10))
        ib.pack_propagate(False)
        tk.Label(ib, text=icon, font=('Helvetica', 12),
                 bg=self.c['blue_bg']).pack(expand=True)

        col = tk.Frame(row, bg=self.c['surface'])
        col.pack(side='left')
        tk.Label(col, text=label,
                 font=('Helvetica', 7),
                 bg=self.c['surface'], fg=self.c['ink4']).pack(anchor='w')
        v = tk.Label(col, text=val,
                     font=('Helvetica', 11, 'bold'),
                     bg=self.c['surface'], fg=self.c['ink'])
        v.pack(anchor='w')
        return v

    # ── RIGHT COLUMN ────────────────────────────────────────

    def _build_right(self, p):

        # ── CHAT CARD (expands to fill space) ───────────────
        cc = self._card(p)
        cc.pack(side='top', fill='both', expand=True, pady=(0, 12))

        # Chat header
        hdr = tk.Frame(cc, bg=self.c['surface'])
        hdr.pack(fill='x', padx=18, pady=(14, 0))

        tk.Label(hdr, text='Conversation',
                 font=('Georgia', 13, 'bold'),
                 bg=self.c['surface'], fg=self.c['ink']).pack(side='left')

        self.count_lbl = tk.Label(hdr, text='',
                                  font=('Helvetica', 8),
                                  bg=self.c['bg'], fg=self.c['ink3'],
                                  padx=8, pady=2)
        self.count_lbl.pack(side='right')

        tk.Frame(cc, bg=self.c['border'], height=1).pack(
            fill='x', padx=18, pady=(10, 0))

        # Scrollable text
        scr = tk.Scrollbar(cc, relief='flat', width=4,
                           bg=self.c['bg'], troughcolor=self.c['bg'])
        scr.pack(side='right', fill='y', padx=(0, 2), pady=4)

        self.chat_box = tk.Text(
            cc,
            bg=self.c['surface'], fg=self.c['ink2'],
            font=('Helvetica', 11),
            wrap='word', state='disabled', relief='flat',
            padx=18, pady=10,
            yscrollcommand=scr.set,
            cursor='arrow', highlightthickness=0,
            spacing1=2, spacing3=6)
        self.chat_box.pack(side='left', fill='both', expand=True)
        scr.config(command=self.chat_box.yview)

        self.chat_box.tag_config('r_lbl', foreground=self.c['blue'],
                                 font=('Helvetica', 8, 'bold'))
        self.chat_box.tag_config('u_lbl', foreground=self.c['ink4'],
                                 font=('Helvetica', 8, 'bold'))
        self.chat_box.tag_config('r_msg', foreground=self.c['ink'],
                                 font=('Helvetica', 11),
                                 lmargin1=12, lmargin2=12)
        self.chat_box.tag_config('u_msg', foreground=self.c['ink2'],
                                 font=('Helvetica', 11),
                                 lmargin1=12, lmargin2=12)

        # ── INPUT ROW (always at bottom, fixed height) ───────
        inp = tk.Frame(p, bg=self.c['bg'])
        inp.pack(side='bottom', fill='x')

        # NAME card
        nc = self._card(inp)
        nc.pack(side='left', fill='y', padx=(0, 12), ipadx=0)

        tk.Label(nc, text='YOUR NAME',
                 font=('Helvetica', 8, 'bold'),
                 bg=self.c['surface'], fg=self.c['ink4']).pack(
                     anchor='w', padx=14, pady=(12, 2))
        tk.Label(nc, text='Type if camera missed you',
                 font=('Helvetica', 7),
                 bg=self.c['surface'], fg=self.c['ink4']).pack(
                     anchor='w', padx=14)

        nf = tk.Frame(nc, bg=self.c['border'], padx=1, pady=1)
        nf.pack(fill='x', padx=14, pady=(6, 0))

        self.name_entry = tk.Entry(
            nf,
            bg=self.c['bg'], fg=self.c['ink3'],
            font=('Helvetica', 12), relief='flat',
            insertbackground=self.c['blue'],
            width=22)
        self.name_entry.pack(fill='x', ipady=9, ipadx=8)
        self._ph_set(self.name_entry, 'e.g.  Malak')
        self.name_entry.bind('<FocusIn>',
            lambda e: self._ph_in(self.name_entry, 'e.g.  Malak'))
        self.name_entry.bind('<FocusOut>',
            lambda e: self._ph_out(self.name_entry, 'e.g.  Malak'))
        self.name_entry.bind('<Return>', self._submit_name)

        tk.Button(nc, text='Save Name',
                  font=('Helvetica', 10, 'bold'),
                  bg=self.c['blue'], fg='white',
                  relief='flat', pady=8, cursor='hand2', bd=0,
                  command=self._submit_name,
                  activebackground=self.c['blue2'],
                  activeforeground='white').pack(
                      fill='x', padx=14, pady=10)

        # MESSAGE card
        mc = self._card(inp)
        mc.pack(side='left', fill='both', expand=True)

        tk.Label(mc, text='SEND A MESSAGE',
                 font=('Helvetica', 8, 'bold'),
                 bg=self.c['surface'], fg=self.c['ink4']).pack(
                     anchor='w', padx=14, pady=(12, 2))
        tk.Label(mc, text='Ask ARIA anything — powered by Mistral AI',
                 font=('Helvetica', 7),
                 bg=self.c['surface'], fg=self.c['ink4']).pack(
                     anchor='w', padx=14)

        mf = tk.Frame(mc, bg=self.c['border'], padx=1, pady=1)
        mf.pack(fill='x', padx=14, pady=(6, 0))

        self.msg_entry = tk.Entry(
            mf,
            bg=self.c['bg'], fg=self.c['ink3'],
            font=('Helvetica', 12), relief='flat',
            insertbackground=self.c['blue'])
        self.msg_entry.pack(fill='x', ipady=9, ipadx=8)

        ph = 'Ask me anything…'
        self._ph_set(self.msg_entry, ph)
        self.msg_entry.bind('<FocusIn>',
            lambda e: self._ph_in(self.msg_entry, ph))
        self.msg_entry.bind('<FocusOut>',
            lambda e: self._ph_out(self.msg_entry, ph))
        self.msg_entry.bind('<Return>', self._submit_msg)

        tk.Button(mc, text='Send  →',
                  font=('Helvetica', 10, 'bold'),
                  bg=self.c['blue'], fg='white',
                  relief='flat', pady=8, cursor='hand2', bd=0,
                  command=self._submit_msg,
                  activebackground=self.c['blue2'],
                  activeforeground='white').pack(
                      fill='x', padx=14, pady=10)

    # ═══════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════

    def _card(self, parent):
        return tk.Frame(parent, bg=self.c['surface'],
                        highlightbackground=self.c['border'],
                        highlightthickness=1)

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

    def _submit_name(self, _=None):
        v = self.name_entry.get().strip()
        if not v or v.startswith('e.g.'):
            return
        self._add_msg('USER', v)
        self.command_queue.put({'type': 'SET_NAME', 'name': v})
        self.lbl_name.config(text=v)
        self.name_entry.delete(0, 'end')

    def _submit_msg(self, _=None):
        ph = 'Ask me anything…'
        v  = self.msg_entry.get().strip()
        if not v or v == ph:
            return
        self._add_msg('USER', v)
        self.command_queue.put({'type': 'USER_MESSAGE', 'text': v})
        self.msg_entry.delete(0, 'end')

    # ── Chat ─────────────────────────────────────────────────

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
        self.count_lbl.config(
            text=f'{n} message{"s" if n != 1 else ""}')

    def add_chat_message(self, sender, msg):
        """Called by other nodes to add messages"""
        self.root.after(0, lambda s=sender, m=msg: self._add_msg(s, m))

    def update_visitor_info(self, name=None, visits=None, mood=None):
        """Called by face recognition / emotion nodes"""
        def _do():
            if name:   self.lbl_name.config(text=name)
            if visits: self.lbl_visits.config(text=str(visits))
            if mood:   self.lbl_mood.config(text=mood.capitalize())
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
    #  FACE
    # ═══════════════════════════════════════════════════════

    def _blink_loop(self):
        import random
        while self.running:
            time.sleep(random.uniform(3, 7))
            self.blink_open = False
            self.root.after(0, lambda: self._redraw(self.current_emotion))
            time.sleep(0.09)
            self.blink_open = True
            self.root.after(0, lambda: self._redraw(self.current_emotion))

    def _float_loop(self):
        while self.running:
            self.anim_t += 0.05
            self.anim_y = int(math.sin(self.anim_t) * 3)
            self.root.after(0, lambda: self._redraw(self.current_emotion))
            time.sleep(0.06)

    def _redraw(self, emotion):
        self.canvas.delete('all')
        col = self.c.get(emotion, self.c['blue'])
        bg  = self.c['face_bg']
        cx  = 100
        cy  = 100 + self.anim_y

        # Glow ring
        self.canvas.create_oval(
            cx-88, cy-88, cx+88, cy+88,
            outline=self._mix(col, '#FFFFFF', 0.65),
            width=10, fill=bg)
        # Head circle
        self.canvas.create_oval(
            cx-78, cy-78, cx+78, cy+78,
            outline=col, width=2, fill=bg)

        self._draw_eyes(cx, cy, col, emotion)
        self._draw_mouth(cx, cy, col, emotion)

        # Nose dot
        self.canvas.create_oval(
            cx-3, cy+5, cx+3, cy+11,
            fill=self._mix(col, '#FFFFFF', 0.5), outline='')

        # Update pill and status
        info = {
            'NEUTRAL':   ('STANDBY',   'Awaiting visitor'),
            'HAPPY':     ('HAPPY',     'Glad to see you!'),
            'SAD':       ('CONCERNED', 'Here to help.'),
            'ANGRY':     ('ALERT',     'Let me resolve this.'),
            'SURPRISED': ('SURPRISED', 'That is unexpected!'),
            'CONFUSED':  ('THINKING',  'Processing your request…'),
            'TIRED':     ('RESTING',   'Brief pause.'),
            'EXCITED':   ('ENGAGED',   'Happy to assist!'),
            'ATTENTIVE': ('LISTENING', 'I am listening.'),
        }
        pill, status = info.get(emotion, ('STANDBY', 'Ready'))
        light = self._mix(col, '#FFFFFF', 0.86)
        self.emotion_pill.config(text=pill, bg=light, fg=col)
        self.face_status.config(text=status)

        moods = {
            'NEUTRAL': 'Neutral',   'HAPPY': 'Happy',
            'SAD': 'Sad',           'ANGRY': 'Concerned',
            'SURPRISED': 'Surprised', 'CONFUSED': 'Thinking',
            'TIRED': 'Resting',     'EXCITED': 'Excited',
            'ATTENTIVE': 'Attentive',
        }
        self.lbl_mood.config(text=moods.get(emotion, '—'))

    def _draw_eyes(self, cx, cy, col, emotion):
        ey = cy - 22
        lx, rx = cx - 26, cx + 26
        r = 13

        if not self.blink_open:
            for ex in (lx, rx):
                self.canvas.create_line(
                    ex-r, ey, ex+r, ey,
                    fill=col, width=3, capstyle='round')
            return

        if emotion in ('HAPPY', 'EXCITED'):
            for ex in (lx, rx):
                self.canvas.create_arc(
                    ex-r, ey-r, ex+r, ey+r,
                    start=0, extent=180,
                    style='arc', outline=col, width=3)

        elif emotion == 'SAD':
            for ex in (lx, rx):
                self.canvas.create_arc(
                    ex-r, ey-r, ex+r, ey+r,
                    start=180, extent=180,
                    style='arc', outline=col, width=3)
            self.canvas.create_line(
                lx-r, ey-18, lx+r, ey-10, fill=col, width=2)
            self.canvas.create_line(
                rx-r, ey-10, rx+r, ey-18, fill=col, width=2)

        elif emotion == 'ANGRY':
            self.canvas.create_line(
                lx-r, ey-16, lx+r, ey-6, fill=col, width=3)
            self.canvas.create_line(
                rx-r, ey-6, rx+r, ey-16, fill=col, width=3)
            for ex in (lx, rx):
                self.canvas.create_rectangle(
                    ex-r, ey-3, ex+r, ey+r,
                    fill=col, outline='')

        elif emotion == 'SURPRISED':
            for ex in (lx, rx):
                self.canvas.create_oval(
                    ex-r, ey-r, ex+r, ey+r,
                    outline=col, width=2, fill='white')
                self.canvas.create_oval(
                    ex-5, ey-5, ex+5, ey+5,
                    fill=col, outline='')

        elif emotion == 'TIRED':
            for ex in (lx, rx):
                self.canvas.create_arc(
                    ex-r, ey-r, ex+r, ey+r,
                    start=0, extent=180,
                    fill=col, outline='')

        elif emotion == 'CONFUSED':
            self.canvas.create_oval(
                lx-r, ey-r, lx+r, ey+r,
                outline=col, width=2, fill='white')
            self.canvas.create_oval(
                lx-5, ey-5, lx+5, ey+5, fill=col, outline='')
            self.canvas.create_arc(
                rx-r, ey-r-4, rx+r, ey+r-4,
                start=0, extent=180,
                style='arc', outline=col, width=3)

        else:
            for ex in (lx, rx):
                self.canvas.create_oval(
                    ex-r, ey-r, ex+r, ey+r,
                    outline=col, width=2, fill='white')
                self.canvas.create_oval(
                    ex-5, ey-5, ex+5, ey+5,
                    fill=col, outline='')
                self.canvas.create_oval(
                    ex-2, ey-4, ex+1, ey-1,
                    fill='white', outline='')

    def _draw_mouth(self, cx, cy, col, emotion):
        my = cy + 34

        if emotion in ('HAPPY', 'EXCITED'):
            self.canvas.create_arc(
                cx-30, my-6, cx+30, my+20,
                start=0, extent=-180,
                style='arc', outline=col, width=3)

        elif emotion == 'SAD':
            self.canvas.create_arc(
                cx-26, my+6, cx+26, my+26,
                start=0, extent=180,
                style='arc', outline=col, width=3)

        elif emotion == 'ANGRY':
            self.canvas.create_line(
                cx-24, my+10, cx+24, my+10,
                fill=col, width=3, capstyle='round')

        elif emotion == 'SURPRISED':
            self.canvas.create_oval(
                cx-10, my+2, cx+10, my+20,
                fill='white', outline=col, width=2)

        elif emotion == 'CONFUSED':
            pts = [cx-24, my+8, cx-12, my+2,
                   cx, my+12, cx+12, my+2, cx+24, my+8]
            self.canvas.create_line(
                pts, fill=col, width=2, smooth=True, capstyle='round')

        elif emotion == 'TIRED':
            self.canvas.create_arc(
                cx-18, my+8, cx+18, my+22,
                start=0, extent=180,
                style='arc', outline=col, width=2)

        else:
            self.canvas.create_arc(
                cx-22, my-2, cx+22, my+14,
                start=0, extent=-180,
                style='arc', outline=col, width=2)

    def _mix(self, c1, c2, t):
        try:
            def p(h):
                h = h.lstrip('#')
                return int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
            r1, g1, b1 = p(c1)
            r2, g2, b2 = p(c2)
            return '#{:02x}{:02x}{:02x}'.format(
                int(r1 + (r2-r1)*t),
                int(g1 + (g2-g1)*t),
                int(b1 + (b2-b1)*t))
        except Exception:
            return c1

    # ═══════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════

    def run(self):
        self.root.mainloop()

    def shutdown(self):
        self.running = False
        try:
            self.root.quit()
        except Exception:
            pass
