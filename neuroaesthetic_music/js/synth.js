/**
 * GestureSynth — plays gestures from the gesture library JSON format.
 *
 * Each gesture has:
 *   bpm: float
 *   events: [{
 *     frequency: Hz,
 *     beats: duration in beats,
 *     amplitude: [0-1],
 *     attack: seconds,
 *     release: seconds,
 *     partials: [w1, w2, ..., w16],   // harmonic weight array
 *     chord: { enabled, chord_frequencies: [Hz, ...], balance: float }
 *   }, ...]
 *
 * Usage:
 *   const synth = new GestureSynth();
 *   synth.playGesture(gestureObject);   // from gestures.json
 *   synth.stop();
 *   synth.getAnalyser();  // AnalyserNode for oscilloscope canvas
 */
class GestureSynth {
  constructor() {
    this.ctx       = null;
    this.nodes     = [];   // all active audio nodes
    this.analyser  = null;
    this._timeouts = [];   // clearTimeout handles for sequential events
  }

  _ensureContext() {
    if (!this.ctx) {
      this.ctx      = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.connect(this.ctx.destination);
    } else if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  /**
   * Play a single note event starting at audioTime.
   * @param {object} event       - NoteEvent in web format
   * @param {float}  audioTime   - ctx.currentTime offset to start
   * @param {float}  durationSec - event duration in seconds
   */
  _playEvent(event, audioTime, durationSec) {
    if (!event.frequency || event.amplitude === 0) return;

    // Collect all voices: root + chord voices if enabled
    const voices = [event.frequency];
    if (event.chord?.enabled && Array.isArray(event.chord.chord_frequencies)) {
      event.chord.chord_frequencies.forEach(f => voices.push(f));
    }

    const releaseTime = Math.min(event.release, durationSec * 0.8);
    const attackTime  = Math.min(event.attack, durationSec * 0.3);

    voices.forEach((baseFreq, voiceIdx) => {
      // Scale amplitude for chord voices (each successive voice quieter)
      const balance  = event.chord?.balance ?? 0.7;
      const voiceAmp = event.amplitude * Math.pow(balance, voiceIdx);

      // Master envelope gain for this voice
      const masterGain = this.ctx.createGain();
      masterGain.gain.setValueAtTime(0, audioTime);
      masterGain.gain.linearRampToValueAtTime(voiceAmp, audioTime + attackTime);
      masterGain.gain.setValueAtTime(voiceAmp, audioTime + durationSec - releaseTime);
      masterGain.gain.linearRampToValueAtTime(0, audioTime + durationSec);
      masterGain.connect(this.analyser);
      this.nodes.push(masterGain);

      // One oscillator per non-zero partial (up to 16 harmonics)
      (event.partials || []).forEach((weight, pIdx) => {
        if (weight < 0.005) return;  // skip near-silent partials
        const freq = baseFreq * (pIdx + 1);
        if (freq > 20000) return;    // skip above hearing range

        const osc  = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type            = 'sine';
        osc.frequency.value = freq;
        gain.gain.value     = weight;

        osc.connect(gain);
        gain.connect(masterGain);
        osc.start(audioTime);
        osc.stop(audioTime + durationSec + 0.05);
        this.nodes.push(osc, gain);
      });
    });
  }

  /**
   * Play a full gesture (sequence of events).
   * @param {object} gesture - single entry from gestures.json
   */
  playGesture(gesture) {
    this._ensureContext();
    this.stop();

    const beatDuration = 60 / gesture.bpm;
    let timeOffset = 0;

    (gesture.events || []).forEach(event => {
      const durationSec = event.beats * beatDuration;
      const startTime   = this.ctx.currentTime + timeOffset;
      this._playEvent(event, startTime, durationSec);
      timeOffset += durationSec;
    });
  }

  /**
   * Total playback duration of a gesture in seconds.
   * @param {object} gesture - single entry from gestures.json
   * @returns {number} duration in seconds
   */
  gestureDuration(gesture) {
    const beatDuration = 60 / gesture.bpm;
    return (gesture.events || []).reduce((s, e) => s + e.beats * beatDuration, 0);
  }

  stop() {
    this._timeouts.forEach(clearTimeout);
    this._timeouts = [];
    this.nodes.forEach(n => {
      try { n.stop?.(); } catch (_) {}
      try { n.disconnect(); } catch (_) {}
    });
    this.nodes = [];
  }

  getAnalyser() { return this.analyser; }
}

window.GestureSynth = GestureSynth;
