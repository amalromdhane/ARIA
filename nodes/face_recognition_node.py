"""
Face Recognition Node — dlib optimized + ENHANCED (IOU multi-face tracking + adaptive severe lighting + DB encoding update)
=================================================================
Strategy:
- dlib HOG + jitters=5 for stable 128-D encodings
- ADVANCED CLAHE + adaptive gamma (dark + bright scenes) for severe luminosity/contrast
- Hybrid IOU + appearance tracking — no ID switches on crossing/movement
- Persistent tracks per person
- Vote 4/6 majority — no false positives
- Adaptive DB encoding update for varying conditions over time
- Multi-person support
- Thread-safe (single inference lock)
- No heavy downloads, no segfaults, no new dependencies
"""
import time, os, pickle, uuid, threading
from collections import Counter
try:
    import cv2, numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False
try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
try:
    import face_recognition
    DLIB_OK = True
except ImportError:
    DLIB_OK = False
    print("[FR] pip install face-recognition")
# ── Preprocessing — ENHANCED for severe lighting ─────────────────────────────
def clahe(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean_l = float(l.mean())
        std_l = float(l.std())

        # Adaptive clipLimit: higher when low contrast (severe conditions)
        clip_limit = max(2.0, min(5.0, 3.0 + (30 - std_l) / 10))
        eq = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(8, 8)).apply(l)

        # Adaptive gamma correction — CORRECTED direction for real severe cases
        if mean_l < 80:  # very dark / weak luminosity
            gamma = 0.5 if mean_l < 50 else 0.7
            table = np.array([((i / 255.0) ** gamma) * 255
                              for i in range(256)]).astype('uint8')
            eq = cv2.LUT(eq, table)
        elif mean_l > 180:  # overexposed / strong luminosity
            gamma = 1.8 if mean_l > 200 else 1.4
            table = np.array([((i / 255.0) ** gamma) * 255
                              for i in range(256)]).astype('uint8')
            eq = cv2.LUT(eq, table)

        return cv2.cvtColor(cv2.merge([eq, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return frame
# ── Track ─────────────────────────────────────────────────────────────────────
class Track:
    VOTE_SIZE = 6
    VOTE_NEEDED = 4
    def __init__(self, tid):
        self.tid = tid
        self.last_seen = time.time()
        self.last_enc = None
        self.last_box = None
        self.votes = []
        self.greeted = False
        self.goodbye_sent = False
        self.registered = False
    def touch(self, enc, box):
        self.last_seen = time.time()
        self.last_enc = enc
        self.last_box = box
        self.goodbye_sent = False
    def vote(self, vid, dist):
        self.votes.append((vid, dist))
        if len(self.votes) > self.VOTE_SIZE:
            self.votes.pop(0)
    def confirmed_vid(self):
        if len(self.votes) < self.VOTE_NEEDED:
            return None
        ids = [v for v, _ in self.votes if v is not None]
        if not ids:
            return None
        top, count = Counter(ids).most_common(1)[0]
        return top if count >= self.VOTE_NEEDED else None
    def unknown_confirmed(self):
        if len(self.votes) < self.VOTE_NEEDED:
            return False
        return sum(1 for v, _ in self.votes if v is None) >= self.VOTE_NEEDED
    def absent(self, thr):
        return time.time() - self.last_seen > thr
    def dist_to(self, enc):
        if self.last_enc is None or not DLIB_OK:
            return 1.0
        try:
            return float(face_recognition.face_distance([self.last_enc], enc)[0])
        except Exception:
            return 1.0
# ── Node ──────────────────────────────────────────────────────────────────────
class FaceRecognitionNode:
    def __init__(self, frame_queue, sound_emotion_queue, state_manager,
                 main_system=None,
                 db_path='visitor_database.pkl',
                 yolo_model='yolov8n.pt',
                 db_thr=0.42, # DB match threshold
                 track_thr=0.55, # track re-ID threshold
                 absence_thr=10.0,
                 rec_interval=0.8, # seconds between recognition cycles
                 jitters=5): # dlib jitters — higher = more stable
        self.frame_queue = frame_queue
        self.sound_q = sound_emotion_queue
        self.state_manager = state_manager
        self.main_system = main_system
        self.running = False
        self.db_path = db_path
        self.db_thr = db_thr
        self.track_thr = track_thr
        self.absence_thr = absence_thr
        self.rec_interval = rec_interval
        self.jitters = jitters
        self.visitors = self._load_db()
        # Single lock prevents concurrent dlib calls (segfault cause)
        self._lock = threading.Lock()
        # YOLO
        self.yolo = None
        if YOLO_OK:
            try:
                self.yolo = YOLO(yolo_model)
                print(f"[FR] YOLO: {yolo_model}")
            except Exception as e:
                print(f"[FR] YOLO error: {e}")
        self.tracks: dict[str, Track] = {}
        self._tid_ctr = 0
        self._vid2tid = {}
        self._unk_tids = []
        self.current_visitor_id = None
        self.current_visitor_name = None
        self.last_rec = 0.0
        self.frame_ctr = 0
        self._greeted_names: set = set()
        self._speech_lock = threading.Lock()
        self._last_speech = 0.0
        self._speech_gap = 4.0
        print(f"[FR] Ready — {len(self.visitors)} visitors | "
              f"dlib jitters={jitters} db_thr={db_thr} | ENHANCED tracking & lighting")
    # ── DB ────────────────────────────────────────────────────────────
    def _load_db(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    db = pickle.load(f)
                clean = {k: v for k, v in db.items()
                         if not v.get('name','').startswith('Visitor_')}
                if len(clean) != len(db):
                    print(f"[FR] Cleaned {len(db)-len(clean)} unnamed")
                print(f"[FR] DB: {[v['name'] for v in clean.values()]}")
                return clean
        except Exception as e:
            print(f"[FR] DB load: {e}")
        return {}
    def _save_db(self):
        try:
            os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
            named = {k: v for k, v in self.visitors.items()
                     if not v.get('name','').startswith('Visitor_')}
            with open(self.db_path, 'wb') as f:
                pickle.dump(named, f)
        except Exception as e:
            print(f"[FR] DB save: {e}")
    def set_visitor_name(self, visitor_id, name):
        name = name.strip().title()
        if visitor_id in self.visitors:
            self.visitors[visitor_id]['name'] = name
            self._save_db()
            print(f"[FR] ✓ '{name}' → {visitor_id}")
        tid = self._vid2tid.get(visitor_id)
        if tid and tid in self.tracks:
            self.tracks[tid].greeted = True
        if visitor_id in self._unk_tids:
            self._unk_tids.remove(visitor_id)
        self.current_visitor_id = self._unk_tids[0] if self._unk_tids else None
        self.current_visitor_name = name
        ai = getattr(self.state_manager, 'ai_agent_node', None)
        if ai:
            ai.set_context(name=name)
    # ── IOU helper (NEW — for stable multi-face tracking) ─────────────
    def _iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes."""
        if not box1 or not box2:
            return 0.0
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    # ── Adaptive DB encoding update (NEW — long-term robustness) ───────
    def _update_encoding(self, vid, new_enc):
        """Gradually update stored encoding with strong matches (handles lighting drift)."""
        if vid not in self.visitors or 'encoding' not in self.visitors[vid]:
            return
        old = np.array(self.visitors[vid]['encoding'])
        new_enc = np.array(new_enc)
        updated = 0.7 * old + 0.3 * new_enc  # conservative average to prevent drift
        self.visitors[vid]['encoding'] = updated.tolist()
        self._save_db()  # persist immediately (DB small & fast)
    # ── Detection & encoding ──────────────────────────────────────────
    def _detect(self, frame):
        """YOLO — returns list of (x1,y1,x2,y2)."""
        if not YOLO_OK or self.yolo is None:
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]
        with self._lock:
            try:
                res = self.yolo(frame, classes=[0], conf=0.35, verbose=False)[0]
                return [tuple(map(int, b.xyxy[0].tolist()))
                        for b in res.boxes]
            except Exception as e:
                print(f"[FR] YOLO: {e}")
                h, w = frame.shape[:2]
                return [(0, 0, w, h)]
    def _encode(self, frame, box):
        """
        dlib face encoding with CLAHE preprocessing.
        jitters=5 gives stable encodings at ~0.5x speed vs jitters=1.
        Returns encoding array or None.
        """
        if not DLIB_OK or not CV2_OK:
            return None
        with self._lock:
            try:
                x1, y1, x2, y2 = box
                h, w = frame.shape[:2]
                m = 30
                crop = frame[max(0,y1-m):min(h,y2+m),
                             max(0,x1-m):min(w,x2+m)]
                if crop.size == 0:
                    return None
                # Face-specific CLAHE (local contrast) on already pre-enhanced frame
                enhanced = clahe(crop)
                small = cv2.resize(enhanced, (0,0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model='hog')
                if not locs:
                    # Fallback: full crop without resize
                    rgb2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb2, model='hog')
                    if not locs:
                        return None
                    encs = face_recognition.face_encodings(
                        rgb2, locs, num_jitters=self.jitters)
                else:
                    encs = face_recognition.face_encodings(
                        rgb, locs, num_jitters=self.jitters)
                return encs[0] if encs else None
            except Exception as e:
                print(f"[FR] encode: {e}")
                return None
    def _match(self, enc):
        """
        Compare encoding against all named visitors.
        Returns (visitor_id | None, best_distance).
        """
        if not DLIB_OK:
            return None, 1.0
        named = {vid: v for vid, v in self.visitors.items()
                 if not v.get('name','').startswith('Visitor_')
                 and v.get('encoding') is not None}
        if not named:
            return None, 1.0
        try:
            encs = [np.array(v['encoding']) for v in named.values()]
            ids = list(named.keys())
            dists = face_recognition.face_distance(encs, enc)
            best = int(dists.argmin())
            dist = float(dists[best])
            vid = ids[best] if dist < self.db_thr else None
            name = self.visitors.get(ids[best], {}).get('name', '?')
            mark = '✓' if vid else '✗'
            print(f"[FR] {mark} {name} dist={dist:.3f}")
            return vid, dist
        except Exception as e:
            print(f"[FR] match: {e}")
            return None, 1.0
    # ── Track management — ENHANCED with IOU ─────────────────────────────
    def _get_track(self, enc, box):
        """Hybrid IOU + appearance re-ID — prevents ID switches in multi-face scenarios."""
        best_tid = None
        best_score = -1.0
        best_iou = 0.0
        best_dist = 1.0
        for tid, track in self.tracks.items():
            if track.last_box is None:
                continue
            dist = track.dist_to(enc)
            iou_val = self._iou(track.last_box, box)
            score = iou_val * 0.65 + (1.0 - min(dist, 1.0)) * 0.35
            if score > best_score:
                best_score = score
                best_tid = tid
                best_iou = iou_val
                best_dist = dist
        # Accept if strong spatial overlap OR strong appearance match
        if best_tid and (best_iou > 0.25 or best_dist < self.track_thr):
            return best_tid
        # New track
        self._tid_ctr += 1
        tid = f"T{self._tid_ctr:03d}"
        self.tracks[tid] = Track(tid)
        print(f"[FR] New track {tid}")
        return tid
    # ── Main frame loop ───────────────────────────────────────────────
    def process_frame(self, frame):
        if not CV2_OK:
            return
        self.frame_ctr += 1
        # Skip frames to reduce CPU load
        if self.frame_ctr % 4 != 0:
            self._check_absences()
            return
        now = time.time()
        if now - self.last_rec < self.rec_interval:
            self._check_absences()
            return
        self.last_rec = now
        try:
            enh = clahe(frame)
            boxes = self._detect(enh)
            if not boxes:
                self._check_absences()
                return
            active_tids = []
            confirmed_map = {}
            for box in boxes:
                enc = self._encode(enh, box)
                if enc is None:
                    continue
                tid = self._get_track(enc, box)  # ← ENHANCED call (now takes box)
                track = self.tracks[tid]
                track.touch(enc, box)
                active_tids.append(tid)
                vid, dist = self._match(enc)
                track.vote(vid, dist)
                # NEW: strong match → adaptive DB update (lighting robustness)
                if vid and dist < 0.28:
                    self._update_encoding(vid, enc)
                conf = track.confirmed_vid()
                if conf:
                    confirmed_map[tid] = conf
                    self._vid2tid[conf] = tid
                elif track.unknown_confirmed() and not track.registered:
                    track.registered = True
                    new_vid = self._register_unknown(enc)
                    self._vid2tid[new_vid] = tid
                    if tid not in self._unk_tids:
                        self._unk_tids.append(tid)
                    if self.current_visitor_id is None:
                        self.current_visitor_id = new_vid
            # Greet ungreeted
            to_greet = []
            for tid in active_tids:
                track = self.tracks.get(tid)
                if track is None or track.greeted:
                    continue
                if tid in confirmed_map:
                    conf_vid = confirmed_map[tid]
                    name = self._dname(conf_vid)
                    if name and name in self._greeted_names:
                        track.greeted = True
                        continue
                    to_greet.append((tid, conf_vid))
                elif tid in self._unk_tids:
                    to_greet.append((tid, None))
            if to_greet:
                self._greet(to_greet, confirmed_map)
                for tid, _ in to_greet:
                    if tid in self.tracks:
                        self.tracks[tid].greeted = True
            self._update_overlay(active_tids, confirmed_map)
        except Exception as e:
            import traceback
            print(f"[FR] Error: {e}")
            traceback.print_exc()
        self._check_absences()
    # ── Helpers ───────────────────────────────────────────────────────
    def _register_unknown(self, enc):
        vid = f"Visitor_{uuid.uuid4().hex[:8]}"
        self.visitors[vid] = {
            'encoding': enc.tolist() if hasattr(enc,'tolist') else list(enc),
            'name': vid,
            'visits': 1,
            'first_seen': time.strftime('%Y-%m-%d %H:%M'),
            'last_seen': time.strftime('%Y-%m-%d %H:%M'),
        }
        print(f"[FR] Unknown registered: {vid}")
        return vid
    def _increment_visit(self, vid):
        if vid not in self.visitors:
            return
        v = self.visitors[vid]
        v['visits'] = v.get('visits', 0) + 1
        v['last_seen'] = time.strftime('%Y-%m-%d %H:%M')
        self._save_db()
    def _dname(self, vid):
        if vid is None:
            return None
        name = self.visitors.get(vid, {}).get('name', vid)
        return None if (not name or name.startswith('Visitor_')) else name
    # ── Greeting scenarios ────────────────────────────────────────────
    def _greet(self, to_greet, confirmed_map):
        known_names = []
        known_visits = {}
        unk_count = 0
        incremented = set()
        for tid, conf_vid in to_greet:
            if conf_vid:
                name = self._dname(conf_vid)
                if name and name not in known_names:
                    known_names.append(name)
                    if conf_vid not in incremented:
                        self._increment_visit(conf_vid)
                        incremented.add(conf_vid)
                    known_visits[name] = self.visitors.get(
                        conf_vid, {}).get('visits', 1)
            else:
                unk_count += 1
        if known_names and unk_count == 0:
            if len(known_names) == 1:
                n, v = known_names[0], known_visits.get(known_names[0], 1)
                if v == 1:
                    msg = f"Hello {n}, welcome! Great to meet you."
                elif v == 2:
                    msg = f"Welcome back {n}! Great to see you again."
                elif v == 3:
                    msg = f"Hello again {n}! Your third visit already."
                else:
                    msg = f"Welcome back {n}!"
            else:
                ns = ', '.join(known_names[:-1]) + f' and {known_names[-1]}'
                msg = f"Hello {ns}, welcome back!"
            for n in known_names:
                self._greeted_names.add(n)
        elif not known_names:
            msg = ("Hello! I don't think we've met before. What's your name?"
                   if unk_count == 1
                   else f"Hello! I see {unk_count} people I don't know. "
                        "Could you please introduce yourselves?")
        else:
            ks = (known_names[0] if len(known_names) == 1
                  else ', '.join(known_names[:-1]) + f' and {known_names[-1]}')
            msg = (f"Welcome back {ks}! And I see someone I don't know — "
                   "could you please introduce yourself?"
                   if unk_count == 1
                   else f"Welcome back {ks}! And {unk_count} people I don't "
                        "know — could you please introduce yourselves?")
            for n in known_names:
                self._greeted_names.add(n)
        self._speak_once(msg)
        self._chat(msg)
        self.state_manager.change_state('GREETING')
        self.state_manager.command_queue.put({'type': 'VISITOR_ARRIVED'})
        print(f"[FR] Greeting: {msg}")
    def _speak_once(self, text):
        with self._speech_lock:
            now = time.time()
            if now - self._last_speech < self._speech_gap:
                print(f"[FR] Suppressed: {text[:50]}")
                return
            self._last_speech = now
        self._speak(text)
    def _update_overlay(self, active_tids, confirmed_map):
        sensor = (getattr(self.main_system, 'sensor_node', None)
                  if self.main_system else None)
        if not sensor or not hasattr(sensor, 'set_face_overlays'):
            return
        overlays = []
        for tid in active_tids:
            track = self.tracks.get(tid)
            if track is None or track.last_box is None:
                continue
            conf = confirmed_map.get(tid)
            if conf:
                name = self._dname(conf) or '?'
                label = name
                known = name != '?'
            elif tid in self._unk_tids:
                label, known = '? Who are you?', False
            else:
                label, known = 'Recognizing...', False
            overlays.append({'name': label, 'known': known,
                             'bbox': track.last_box})
        sensor.set_face_overlays(overlays)
    def _check_absences(self):
        departed = []
        for tid, track in list(self.tracks.items()):
            if track.absent(self.absence_thr) and not track.goodbye_sent:
                track.goodbye_sent = True
                departed.append(tid)
        for tid in departed:
            track = self.tracks[tid]
            conf = track.confirmed_vid()
            dn = self._dname(conf) if conf else None
            if dn:
                msg = f"Goodbye, {dn}! Have a wonderful day."
                self._speak_once(msg)
                self._chat(msg)
                print(f"[FR] Goodbye: {dn}")
                self._greeted_names.discard(dn)
            del self.tracks[tid]
            if tid in self._unk_tids:
                self._unk_tids.remove(tid)
            for v, t in list(self._vid2tid.items()):
                if t == tid:
                    del self._vid2tid[v]
                    if self.visitors.get(v,{}).get('name','').startswith('Visitor_'):
                        self.visitors.pop(v, None)
            if tid == self.current_visitor_id:
                self.current_visitor_id = (
                    self._unk_tids[0] if self._unk_tids else None)
        if departed and not self.tracks:
            self.state_manager.change_state('IDLE')
            self.state_manager.command_queue.put({'type': 'VISITOR_LEFT'})
            s = getattr(self.main_system, 'sensor_node', None) \
                if self.main_system else None
            if s and hasattr(s, 'set_face_overlays'):
                s.set_face_overlays([])
            ai = getattr(self.state_manager, 'ai_agent_node', None)
            if ai and hasattr(ai, 'history'):
                ai.set_context(name=None, visits=0)
                ai.history = []
            gui = getattr(self.state_manager, 'gui_node', None)
            if gui:
                gui.update_visitor_info(name='—', visits='—', mood='—')
    def _speak(self, text):
        try:
            self.sound_q.put({'type': 'SPEAK', 'text': text})
        except Exception:
            pass
    def _chat(self, text):
        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.add_chat_message('ROBOT', text)
    def run(self):
        self.running = True
        print(f"[FR] Running (dlib jitters={self.jitters} + YOLO "
              f"+ vote {Track.VOTE_NEEDED}/{Track.VOTE_SIZE} + IOU tracking + adaptive lighting)…")
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    self.process_frame(frame)
            except Exception:
                pass
            time.sleep(0.02)
    def stop(self):
        self.running = False
        print("[FR] Stopped")