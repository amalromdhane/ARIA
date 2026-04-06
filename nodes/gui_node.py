"""
GUI Node — Live camera feed with face detection overlay
- Shows live camera feed instead of animated face
- Draws green rectangle (known) or red (unknown) around detected faces
- Shows name label on rectangle after recognition
- Returns to animated smile when no one is detected
"""

import tkinter as tk
from tkinter import Canvas
import threading
import time
import math
import queue

from PIL import Image, ImageTk
PIL_AVAILABLE = True

import cv2
import numpy as np
CV2_AVAILABLE = True


class GUINode:
    def __init__(self, state_queue, emotion_queue, command_queue):
        self.state_queue   = state_queue
        self.emotion_queue = emotion_queue
        self.command_queue = command_queue

        self.root = tk.Tk()
        self.root.title("ARIA — Reception Assistant")
        self.root.geometry("1280x800+100+100")
        self.root.configure(bg='#0A0E1A')
        self.root.resizable(True, True)
        self.root.withdraw()

        self.current_emotion = 'NEUTRAL'
        self.running         = True
        self.blink_open      = True
        self.anim_t          = 0.0
        self.anim_y          = 0
        self._msg_count      = 0

        # ── Camera feed state ─────────────────────────────────────────
        self._camera_lock    = threading.Lock()
        self._current_frame  = None    # latest BGR frame
        self._face_overlays  = []      # list of {bbox, name, known}
        self._last_frame_t   = 0.0
        self._showing_camera = False
        self._photo_ref      = None    # prevent GC

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
            'red':       '#EF4444',
            'green_bg':  '#064E3B',
            'face_bg':   '#0D1220',
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

        for fn in [self._blink_loop, self._float_loop,
                   self._poll, self._camera_loop]:
            threading.Thread(target=fn, daemon=True).start()

        self.root.after(100, lambda: self._add_msg(
            'ROBOT',
            'Good day! I am ARIA, your reception assistant. '
            'Please look at the camera.'))

    # ── Build UI ──────────────────────────────────────────────────────

    def _build(self):
        r = self.root

        # Top bar
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
        tk.Label(lf, text='Reception Assistant',
                 font=('Helvetica', 9),
                 bg=self.c['surface'], fg=self.c['ink3']).pack(side='left')

        rf = tk.Frame(bar, bg=self.c['surface'])
        rf.pack(side='right', padx=20, pady=10)
        self.clock_lbl = tk.Label(rf, font=('Helvetica', 10),
                                  bg=self.c['surface'], fg=self.c['ink3'])
        self.clock_lbl.pack(side='right', padx=(10, 0))
        self._tick()
        tk.Label(rf, text='● ONLINE', font=('Helvetica', 8, 'bold'),
                 bg=self.c['surface'], fg=self.c['green']).pack(side='right')

        tk.Frame(r, bg=self.c['border'], height=1).pack(side='top', fill='x')

        body = tk.Frame(r, bg=self.c['bg'])
        body.pack(side='top', fill='both', expand=True)

        left = tk.Frame(body, bg=self.c['bg'], width=560)
        left.pack(side='left', fill='y')
        left.pack_propagate(False)
        self._build_left(left)

        tk.Frame(body, bg=self.c['border'], width=1).pack(side='left', fill='y')

        right = tk.Frame(body, bg=self.c['bg'])
        right.pack(side='left', fill='both', expand=True)
        self._build_right(right)

    def _build_left(self, p):
        top = tk.Frame(p, bg=self.c['bg'])
        top.pack(fill='x', padx=20, pady=(16, 0))

        self.emotion_pill = tk.Label(
            top, text='● STANDBY',
            font=('Helvetica', 9, 'bold'),
            bg=self.c['blue_glow'], fg=self.c['blue'],
            padx=14, pady=5)
        self.emotion_pill.pack(side='left')

        self.msg_count_lbl = tk.Label(
            top, text='', font=('Helvetica', 8),
            bg=self.c['bg'], fg=self.c['ink3'])
        self.msg_count_lbl.pack(side='right')

        # Single canvas — shows camera OR animated face
        face_frame = tk.Frame(p, bg=self.c['face_bg'],
                              highlightbackground=self.c['border'],
                              highlightthickness=1)
        face_frame.pack(fill='both', expand=True, padx=16, pady=12)

        self.canvas = Canvas(face_frame,
                             bg=self.c['face_bg'],
                             highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>',
                         lambda e: self._redraw(self.current_emotion))

        self.face_status = tk.Label(
            p, text='Look at the camera',
            font=('Helvetica', 11),
            bg=self.c['bg'], fg=self.c['ink3'])
        self.face_status.pack(pady=(0, 8))

        # Visitor info strip
        vi = tk.Frame(p, bg=self.c['surface2'],
                      highlightbackground=self.c['border'],
                      highlightthickness=1)
        vi.pack(fill='x', padx=16, pady=(0, 16))

        row = tk.Frame(vi, bg=self.c['surface2'])
        row.pack(fill='x', padx=16, pady=10)

        for icon, attr, val in [
            ('👤', 'name',   '—'),
            ('🔁', 'visits', '—'),
            ('💬', 'mood',   '—'),
        ]:
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
        inp = tk.Frame(p, bg=self.c['surface'],
                       highlightbackground=self.c['border'],
                       highlightthickness=1)
        inp.pack(side='bottom', fill='x', padx=16, pady=16)

        tk.Label(inp, text='SEND A MESSAGE',
                 font=('Helvetica', 8, 'bold'),
                 bg=self.c['surface'], fg=self.c['ink3']).pack(
                     anchor='w', padx=14, pady=(12, 2))
        tk.Label(inp,
                 text='Type your name when asked, or any question',
                 font=('Helvetica', 7),
                 bg=self.c['surface'], fg=self.c['ink3']).pack(
                     anchor='w', padx=14)

        entry_row = tk.Frame(inp, bg=self.c['surface'])
        entry_row.pack(fill='x', padx=14, pady=(8, 14))

        ef = tk.Frame(entry_row, bg=self.c['border'], padx=1, pady=1)
        ef.pack(side='left', fill='x', expand=True, padx=(0, 10))

        self.msg_entry = tk.Entry(
            ef, bg=self.c['surface2'], fg=self.c['ink'],
            font=('Helvetica', 13), relief='flat',
            insertbackground=self.c['blue'])
        self.msg_entry.pack(fill='x', ipady=11, ipadx=10)

        ph = 'Type here…'
        self._ph_set(self.msg_entry, ph)
        self.msg_entry.bind('<FocusIn>',
                            lambda e: self._ph_in(self.msg_entry, ph))
        self.msg_entry.bind('<FocusOut>',
                            lambda e: self._ph_out(self.msg_entry, ph))
        self.msg_entry.bind('<Return>', self._submit_msg)

        tk.Button(entry_row, text='Send →',
                  font=('Helvetica', 11, 'bold'),
                  bg=self.c['blue'], fg='white',
                  relief='flat', padx=20, pady=10,
                  cursor='hand2', bd=0,
                  command=self._submit_msg,
                  activebackground=self.c['blue2'],
                  activeforeground='white').pack(side='left')

        cc = tk.Frame(p, bg=self.c['bg'])
        cc.pack(side='top', fill='both', expand=True, padx=16, pady=(16, 0))

        tk.Label(cc, text='Conversation',
                 font=('Georgia', 13, 'bold'),
                 bg=self.c['bg'], fg=self.c['ink']).pack(anchor='w')
        tk.Frame(cc, bg=self.c['border'], height=1).pack(
            fill='x', pady=(4, 4))

        scr = tk.Scrollbar(cc, relief='flat', width=3,
                           bg=self.c['bg'], troughcolor=self.c['bg'])
        scr.pack(side='right', fill='y')

        self.chat_box = tk.Text(
            cc, bg=self.c['bg'], fg=self.c['ink2'],
            font=('Helvetica', 11), wrap='word',
            state='disabled', relief='flat', padx=4, pady=4,
            yscrollcommand=scr.set, cursor='arrow',
            highlightthickness=0, spacing1=2, spacing3=8)
        self.chat_box.pack(side='left', fill='both', expand=True)
        scr.config(command=self.chat_box.yview)

        for tag, fg, font in [
            ('r_lbl', self.c['blue'],  ('Helvetica', 8, 'bold')),
            ('u_lbl', self.c['ink3'],  ('Helvetica', 8, 'bold')),
            ('r_msg', self.c['ink'],   ('Helvetica', 11)),
            ('u_msg', self.c['ink2'],  ('Helvetica', 11)),
        ]:
            self.chat_box.tag_config(
                tag, foreground=fg, font=font,
                lmargin1=8, lmargin2=8)

    # ── Camera feed ───────────────────────────────────────────────────

    def update_camera_frame(self, bgr_frame, face_overlays=None):
        """
        Called every frame by sensor_node or face_recognition_node.
        bgr_frame:    numpy BGR frame from camera
        face_overlays: list of {bbox:(x1,y1,x2,y2), name:str, known:bool}
        """
        with self._camera_lock:
            self._current_frame  = bgr_frame
            self._face_overlays  = face_overlays or []
            self._last_frame_t   = time.time()

    def _camera_loop(self):
        """Background thread — pushes camera frames to canvas at ~20 fps."""
        while self.running:
            try:
                with self._camera_lock:
                    frame    = self._current_frame
                    overlays = list(self._face_overlays)
                    last_t   = self._last_frame_t

                # If no frame for > 3 seconds → show animated face
                if frame is None:
                    if self._showing_camera:
                        self._showing_camera = False
                        self.root.after(0,
                            lambda: self._redraw(self.current_emotion))
                    time.sleep(0.05)
                    continue

                self._showing_camera = True
                self.root.after(0,
                    lambda f=frame, o=overlays: self._draw_camera(f, o))

            except Exception as e:
                print(f"[GUI] camera_loop error: {e}")
            time.sleep(0.05)   # ~20 fps

    def _draw_camera(self, frame, overlays):
        """Draw camera frame + face rectangles on canvas."""
        if not PIL_AVAILABLE or not CV2_AVAILABLE:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        try:
            # Resize frame to fit canvas
            fh, fw = frame.shape[:2]
            scale  = min(w / fw, h / fh)
            nw     = int(fw * scale)
            nh     = int(fh * scale)
            resized = cv2.resize(frame, (nw, nh))

            # Offset to center in canvas
            ox = (w - nw) // 2
            oy = (h - nh) // 2

            # Draw face rectangles on the resized frame
            display = resized.copy()
            for ov in overlays:
                bbox  = ov.get('bbox')
                name  = ov.get('name', '?')
                known = ov.get('known', False)

                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                # Scale bbox to resized frame
                x1s = int(x1 * scale)
                y1s = int(y1 * scale)
                x2s = int(x2 * scale)
                y2s = int(y2 * scale)

                # Color: green=known, red=unknown
                color = (0, 220, 80) if known else (0, 60, 220)

                # Rectangle
                cv2.rectangle(display, (x1s, y1s), (x2s, y2s), color, 2)

                # Label background + text
                label = name if known else '?  Who are you?'
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                label_y = max(0, y1s - th - 8)
                cv2.rectangle(display,
                              (x1s, label_y),
                              (x1s + tw + 8, y1s),
                              color, -1)
                cv2.putText(display, label,
                            (x1s + 4, y1s - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Convert to PhotoImage
            rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img)

            self._photo_ref = photo   # prevent GC

            self.canvas.delete('all')

            # Black background
            self.canvas.create_rectangle(0, 0, w, h,
                                         fill=self.c['face_bg'], outline='')
            # Camera image centered
            self.canvas.create_image(ox, oy, image=photo, anchor='nw')

            # Status text
            if overlays:
                names = [o['name'] for o in overlays if o.get('known')]
                unkn  = sum(1 for o in overlays if not o.get('known'))
                if names and not unkn:
                    status = f"Welcome, {', '.join(names)}!"
                elif names and unkn:
                    status = f"{', '.join(names)} + {unkn} unknown"
                else:
                    status = f"{unkn} unknown face{'s' if unkn>1 else ''}"
            else:
                status = 'Look at the camera'

            self.face_status.config(text=status)

        except Exception as e:
            print(f"[GUI] _draw_camera error: {e}")

    # ── Called by face_recognition_node ──────────────────────────────

    def update_detected_faces(self, faces_data: list):
        """
        faces_data: list of {name, bbox, known}
        bbox = (x1, y1, x2, y2) in original frame coordinates
        """
        overlays = [
            {
                'bbox':  f.get('bbox'),
                'name':  f.get('name', '?'),
                'known': f.get('known', False),
            }
            for f in faces_data if f.get('bbox') is not None
        ]
        with self._camera_lock:
            self._face_overlays = overlays

    def clear_detected_faces(self):
        with self._camera_lock:
            self._face_overlays = []
        # Don't clear _current_frame — camera stays live

    # ── Animated face (shown when no camera feed) ─────────────────────

    def _redraw(self, emotion):
        if self._showing_camera:
            return   # Camera is active — don't draw animated face

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return

        self.canvas.delete('all')
        col = self.c.get(emotion, self.c['blue'])
        bg  = self.c['face_bg']
        cx  = w // 2
        cy  = h // 2 + self.anim_y
        R   = min(w, h) // 2 - 20

        self.canvas.create_oval(cx-R-16, cy-R-16, cx+R+16, cy+R+16,
                                outline=self._mix(col, '#000000', 0.5),
                                width=20, fill=bg)
        self.canvas.create_oval(cx-R-6, cy-R-6, cx+R+6, cy+R+6,
                                outline=self._mix(col, '#000000', 0.3),
                                width=8, fill=bg)
        self.canvas.create_oval(cx-R, cy-R, cx+R, cy+R,
                                outline=col, width=3, fill=bg)

        self._draw_eyes(cx, cy, col, emotion, R)
        self._draw_mouth(cx, cy, col, emotion, R)

        nr = max(4, R // 20)
        self.canvas.create_oval(cx-nr, cy+R//12, cx+nr, cy+R//6,
                                fill=self._mix(col, '#FFFFFF', 0.4),
                                outline='')

        info = {
            'NEUTRAL':   ('● STANDBY',   'Look at the camera'),
            'HAPPY':     ('● HAPPY',     'Glad to see you!'),
            'SAD':       ('● CONCERNED', 'Here to help.'),
            'ANGRY':     ('● ALERT',     'Let me resolve this.'),
            'SURPRISED': ('● SURPRISED', 'That is unexpected!'),
            'CONFUSED':  ('● THINKING',  'Processing…'),
            'TIRED':     ('● RESTING',   'Brief pause.'),
            'EXCITED':   ('● ENGAGED',   'Happy to assist!'),
            'ATTENTIVE': ('● LISTENING', 'I am listening.'),
        }
        pill_txt, status = info.get(emotion, ('● STANDBY', 'Look at the camera'))
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

    # ── Eyes and mouth ────────────────────────────────────────────────

    def _draw_eyes(self, cx, cy, col, emotion, R):
        ey  = cy - R // 3
        off = R // 3
        lx, rx = cx - off, cx + off
        r   = R // 6
        lw  = max(3, R // 28)

        if not self.blink_open:
            for ex in (lx, rx):
                self.canvas.create_line(ex-r, ey, ex+r, ey,
                                        fill=col, width=lw, capstyle='round')
            return

        if emotion in ('HAPPY', 'EXCITED'):
            for ex in (lx, rx):
                self.canvas.create_arc(ex-r, ey-r, ex+r, ey+r,
                                       start=0, extent=180,
                                       style='arc', outline=col, width=lw)
        elif emotion == 'SURPRISED':
            for ex in (lx, rx):
                self.canvas.create_oval(ex-r, ey-r, ex+r, ey+r,
                                        outline=col, width=lw, fill='')
                pr = r // 2
                self.canvas.create_oval(ex-pr, ey-pr, ex+pr, ey+pr,
                                        fill=col, outline='')
        else:
            for ex in (lx, rx):
                self.canvas.create_oval(ex-r, ey-r, ex+r, ey+r,
                                        outline=col, width=lw, fill='')
                pr = r // 2
                self.canvas.create_oval(ex-pr, ey-pr, ex+pr, ey+pr,
                                        fill=col, outline='')

    def _draw_mouth(self, cx, cy, col, emotion, R):
        my = cy + R // 2
        mw = R // 2
        lw = max(3, R // 28)
        if emotion in ('HAPPY', 'EXCITED'):
            self.canvas.create_arc(cx-mw, my-mw//3, cx+mw, my+mw//2,
                                   start=0, extent=-180,
                                   style='arc', outline=col, width=lw)
        elif emotion == 'SURPRISED':
            or_ = mw // 3
            self.canvas.create_oval(cx-or_, my-or_//2, cx+or_, my+or_,
                                    fill='', outline=col, width=lw)
        else:
            self.canvas.create_arc(cx-mw//2, my-mw//6, cx+mw//2, my+mw//4,
                                   start=0, extent=-180,
                                   style='arc', outline=col, width=lw-1)

    # ── Helpers ───────────────────────────────────────────────────────

    def _tick(self):
        self.clock_lbl.config(text=time.strftime('%H:%M'))
        self.root.after(30000, self._tick)

    def _ph_set(self, e, ph):
        e.insert(0, ph); e.config(fg=self.c['ink3'])

    def _ph_in(self, e, ph):
        if e.get() == ph:
            e.delete(0, 'end'); e.config(fg=self.c['ink'])

    def _ph_out(self, e, ph):
        if not e.get():
            e.insert(0, ph); e.config(fg=self.c['ink3'])

    def _submit_msg(self, _=None):
        ph = 'Type here…'
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
        if threading.current_thread() is threading.main_thread():
            self._add_msg(sender, msg)
        else:
            self.root.after(0,
                lambda s=sender, m=msg: self._add_msg(s, m))

    def update_visitor_info(self, name=None, visits=None, mood=None):
        def _do():
            if name   is not None: self.lbl_name.config(text=name)
            if visits is not None: self.lbl_visits.config(text=str(visits))
            if mood   is not None:
                self.lbl_mood.config(
                    text=mood.capitalize() if mood != '—' else '—')
        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            self.root.after(0, _do)

    def _poll(self):
        while self.running:
            try:
                if not self.emotion_queue.empty():
                    emotion = self.emotion_queue.get_nowait()
                    if emotion != self.current_emotion:
                        self.current_emotion = emotion
                        if not self._showing_camera:
                            self.root.after(
                                0, lambda e=emotion: self._redraw(e))
            except Exception:
                pass
            time.sleep(0.05)

    def _blink_loop(self):
        import random
        while self.running:
            time.sleep(random.uniform(2.5, 6))
            if not self._showing_camera:
                self.blink_open = False
                try:
                    self.root.after(0,
                        lambda: self._redraw(self.current_emotion))
                except RuntimeError:
                    return
                time.sleep(0.08)
                self.blink_open = True
                try:
                    self.root.after(0,
                        lambda: self._redraw(self.current_emotion))
                except RuntimeError:
                    return

    def _float_loop(self):
        while self.running:
            self.anim_t += 0.04
            self.anim_y  = int(math.sin(self.anim_t) * 5)
            time.sleep(0.05)

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
        self.running = True
        print("[GUI] Starting mainloop...")
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.update_idletasks()
        self._redraw('NEUTRAL')
        self.root.mainloop()

    def shutdown(self):
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        print("[GUI] Shutdown")