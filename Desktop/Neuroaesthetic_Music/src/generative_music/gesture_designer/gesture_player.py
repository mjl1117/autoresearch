"""
gesture_player.py
OSC playback engine for the Spectral Gesture Designer.

Connects to a running scsynth instance on port 57110.
Handles live preview (sustain a synth while editing) and
full gesture playback (sequence of timed events).

MJL Neuroaesthetic Music Research — 2026
"""
from __future__ import annotations
import time, threading, socket
from typing import Optional, Callable

_NODE_ID_LOCK = threading.Lock()
_node_id_counter = 2000

from pythonosc import udp_client
from .gesture_model import Gesture, NoteEvent, chord_frequencies, apply_inversion
from .dissonance import DissonanceCalculator, DissonanceTracker

SC_SYNTH_PORT = 57110


def _scsynth_reachable() -> bool:
    """Quick check: can we reach scsynth?"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(0.3)
        try:
            s.sendto(b'/status\x00,\x00\x00\x00', ('127.0.0.1', SC_SYNTH_PORT))
            s.recvfrom(256)
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False


class SynthHandle:
    """Thin wrapper around an active scsynth node."""

    def __init__(self, node_id: int, osc: udp_client.SimpleUDPClient):
        self.node_id = node_id
        self._osc    = osc
        self.alive   = True

    def set(self, param: str, value: float):
        if self.alive:
            self._osc.send_message('/n_set', [self.node_id, param, float(value)])

    def set_many(self, params: dict):
        if not self.alive:
            return
        msg = [self.node_id]
        for k, v in params.items():
            msg += [k, float(v)]
        self._osc.send_message('/n_set', msg)

    def release(self, release_time: float = 0.5):
        """Trigger ADSR release stage."""
        self.set('gate', 0)
        self.alive = False

    def free(self):
        if self.alive:
            self._osc.send_message('/n_free', [self.node_id])
            self.alive = False


class GesturePlayer:
    """Manages live preview synths and full-gesture playback.

    Usage:
        player = GesturePlayer()
        player.start_preview(event)   # sustain while editing
        player.update_preview(event)  # re-send params to live synth
        player.stop_preview()
        player.play_gesture(gesture, on_event_cb, on_done_cb)
        player.stop_gesture()
    """

    def __init__(self):
        self._osc:             Optional[udp_client.SimpleUDPClient] = None
        self._preview:         Optional[SynthHandle] = None
        self._preview_handles: list = []   # extra handles for chord preview
        self._play_thread:     Optional[threading.Thread] = None
        self._stop_flag:       threading.Event = threading.Event()
        self._connected:       bool = False

        # Dissonance tracking
        self._diss_calc:    DissonanceCalculator = DissonanceCalculator()
        self._diss_tracker: DissonanceTracker    = DissonanceTracker(window_sec=10.0)
        self._diss_log:     list = []   # [(time_sec, score)] for full-gesture export

        self._connect()

    # ── Connection ───────────────────────────────────────────────────────────

    def _connect(self):
        self._osc = udp_client.SimpleUDPClient('127.0.0.1', SC_SYNTH_PORT)
        self._connected = _scsynth_reachable()

    def reconnect(self) -> bool:
        self._connect()
        return self._connected

    @property
    def connected(self) -> bool:
        return self._connected

    def check_connection(self) -> bool:
        self._connected = _scsynth_reachable()
        return self._connected

    # ── Node allocation ──────────────────────────────────────────────────────

    def _alloc_id(self) -> int:
        global _node_id_counter
        with _NODE_ID_LOCK:
            nid = _node_id_counter
            _node_id_counter = (_node_id_counter + 1) if _node_id_counter < 8000 else 2000
            return nid

    def _new_synth(self, defname: str, params: dict) -> SynthHandle:
        """Send /s_new and return a SynthHandle."""
        nid = self._alloc_id()
        msg = [defname, nid, 0, 0]
        for k, v in params.items():
            msg += [k, float(v)]
        self._osc.send_message('/s_new', msg)
        return SynthHandle(nid, self._osc)

    # ── Build OSC params from NoteEvent ──────────────────────────────────────

    @staticmethod
    def _event_params(event: NoteEvent, freq_override: float = None) -> dict:
        params = {
            'root':       freq_override if freq_override is not None else event.frequency,
            'amp':        event.amplitude,
            'gate':       1,
            'brightness': event.brightness,
        }
        params.update(event.partials.as_osc_dict())
        return params

    # ── Live preview ─────────────────────────────────────────────────────────

    def start_preview(self, event: NoteEvent):
        """Spawn a sustaining synth (or chord voices) for the given event."""
        if not self._connected or event.is_rest:
            return
        self.stop_preview()
        if event.chord.enabled:
            handles = self._play_chord_handles(event)
            self._preview = handles[0] if handles else None
            self._preview_handles = handles[1:]
        else:
            self._preview = self._new_synth('harmonicSeries', self._event_params(event))

    def update_preview(self, event: NoteEvent):
        """Push updated params to the currently sustaining preview synth."""
        if not self._connected or event.is_rest:
            return
        # For chord preview, restart to reflect any voice/type changes
        if event.chord.enabled:
            self.start_preview(event)
            return
        if self._preview is None:
            return
        params = self._event_params(event)
        params.pop('gate', None)
        self._preview.set_many(params)

    def stop_preview(self):
        if self._preview is not None:
            self._preview.release()
            self._preview = None
        for h in self._preview_handles:
            if h.alive:
                h.release()
        self._preview_handles = []

    # ── Full gesture playback ────────────────────────────────────────────────

    # ── Dissonance API ───────────────────────────────────────────────────────

    @property
    def dissonance_arc(self) -> dict:
        """Current rolling dissonance arc (live during playback, final after)."""
        return self._diss_tracker.current_arc()

    def dissonance_log(self) -> list:
        """Full [(time_sec, score)] log from the last completed gesture."""
        return list(self._diss_log)

    # ── Full gesture playback ────────────────────────────────────────────────

    def play_gesture(self,
                     gesture:       Gesture,
                     on_event:      Optional[Callable[[int], None]] = None,
                     on_done:       Optional[Callable[[], None]]    = None,
                     on_dissonance: Optional[Callable[[float, dict], None]] = None):
        """Play the entire gesture in a background thread.

        on_event(index)            fires when each event begins.
        on_done()                  fires when playback finishes or is stopped.
        on_dissonance(score, arc)  fires on every note with the current
                                   dissonance score and rolling arc descriptor.
                                   Use this to drive UI or real-time synthesis
                                   parameter changes.
        """
        if not self._connected:
            if on_done:
                on_done()
            return
        self.stop_gesture()
        self._stop_flag.clear()
        self._diss_tracker.reset()
        self._diss_log.clear()
        self._play_thread = threading.Thread(
            target=self._play_worker,
            args=(gesture, on_event, on_done, on_dissonance),
            daemon=True
        )
        self._play_thread.start()

    def stop_gesture(self):
        self._stop_flag.set()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=1.0)
        self._play_thread = None

    def _play_worker(self,
                     gesture:       Gesture,
                     on_event:      Optional[Callable[[int], None]],
                     on_done:       Optional[Callable[[], None]],
                     on_dissonance: Optional[Callable[[float, dict], None]] = None):
        beat_dur    = gesture.beat_duration()
        active_handles: list = []
        playback_start = time.time()

        for idx, event in enumerate(gesture.events):
            if self._stop_flag.is_set():
                break

            # Fire UI callback
            if on_event:
                on_event(idx)

            # ── Dissonance tracking ──────────────────────────────────────
            t_now = time.time() - playback_start
            if not event.is_rest:
                diss_score = self._diss_calc.note_dissonance(event)
                self._diss_tracker.push(diss_score, t_now)
                self._diss_log.append((t_now, diss_score))
                if on_dissonance:
                    on_dissonance(diss_score, self._diss_tracker.current_arc())
            # ─────────────────────────────────────────────────────────────

            dur = event.beats * beat_dur

            if event.is_rest:
                time.sleep(dur)
                continue

            if event.pulse.enabled:
                self._play_pulse_burst(event, dur)
            elif event.chord.enabled:
                handles = self._play_chord_handles(event)
                active_handles.extend(handles)
                hold = max(0.05, dur - event.release)
                time.sleep(hold)
                if not self._stop_flag.is_set():
                    for h in handles:
                        h.release(event.release)
                    time.sleep(event.release)
            else:
                # Spawn synth, hold for (beats - release) then gate off
                params = self._event_params(event)
                h = self._new_synth('harmonicSeries', params)
                active_handles.append(h)
                hold = max(0.05, dur - event.release)
                time.sleep(hold)
                if not self._stop_flag.is_set():
                    h.release(event.release)
                    time.sleep(event.release)

        # Clean up any lingering synths
        for h in active_handles:
            if h.alive:
                h.free()

        if on_done:
            on_done()

    def _play_chord_handles(self, event: NoteEvent) -> list:
        """Spawn one harmonicSeries synth per chord voice; return all handles."""
        freqs = chord_frequencies(
            event.frequency, event.chord.chord_type, event.chord.num_voices)
        freqs = apply_inversion(freqs, event.chord.inversion)
        handles = []
        for i, freq in enumerate(freqs):
            amp = event.amplitude * (event.chord.balance ** i)
            params = self._event_params(event, freq_override=freq)
            params['amp'] = amp
            handles.append(self._new_synth('harmonicSeries', params))
        return handles

    def _play_pulse_burst(self, event: NoteEvent, total_dur: float):
        """Fire a burst of short synths within one beat slot."""
        print(f"[PULSE BURST] Called! count={event.pulse.count}, total_dur={total_dur}")
        p = event.pulse
        n = max(1, p.count)
        interval = total_dur / n
        
        active_pulses = []
        
        for i in range(n):
            if self._stop_flag.is_set():
                # Clean up active pulses
                for h in active_pulses:
                    if h.alive:
                        h.free()
                return
            
            # Calculate interpolated parameters
            t = i / max(1, n - 1) if n > 1 else 0.0
            brightness = p.brightness_start + t * (p.brightness_end - p.brightness_start)
            amp        = p.amp_start        + t * (p.amp_end        - p.amp_start)
            
            print(f"[PULSE] Firing synth {i+1}/{n}: amp={amp:.3f}, brightness={brightness:.3f}")
            
            # Create synth with CORRECT parameters for harmonicSeries
            params = {
                'root':       event.frequency,     # ✓ Correct parameter name
                'amp':        amp,                 # ✓ Interpolated amplitude
                'gate':       1,                   # ✓ Start with gate open
                'brightness': brightness,          # ✓ Interpolated brightness
            }
            params.update(event.partials.as_osc_dict())  # ✓ Include partial weights
            
            h = self._new_synth('harmonicSeries', params)
            active_pulses.append(h)
            
            # Wait for decay time, then release this pulse
            decay_time = min(p.decay, interval * 0.9)  # Don't overlap too much
            time.sleep(decay_time)
            
            # Release the pulse
            h.release(0.05)  # Quick release (50ms)
            
            # Wait for remainder of interval
            remaining = interval - decay_time
            if remaining > 0:
                time.sleep(remaining)
        
        # Clean up any remaining pulses
        for h in active_pulses:
            if h.alive:
                h.free()

    # ── Free all ─────────────────────────────────────────────────────────────

    def panic(self):
        """Free all nodes on the server (emergency stop).

        Targets group 1 (the default group) rather than group 0 (root).
        This kills all playing synths without destroying the group tree,
        so the server stays healthy for the next tool opened in the session.
        """
        if self._osc:
            self._osc.send_message('/g_freeAll', [1])
        self.stop_gesture()
        self.stop_preview()
