/**
 * DEFECTVISION 2.0 - TELEMETRY, AUDIO SIRENS & KPI ANALYTICS MODULE
 * Generates audio alerts, routing signals, and updates HUD indicators.
 */

const TelemetryManager = {
  audioCtx: null,

  initAudio() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        this.audioCtx = new AudioContext();
      }
    }
  },

  playAlert(isAnomaly) {
    try {
      this.initAudio();
      if (!this.audioCtx) return;

      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }

      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      const now = this.audioCtx.currentTime;

      if (isAnomaly) {
        // Warning siren for Defect (Double Pulsing Beep)
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.exponentialRampToValueAtTime(440, now + 0.15);
        osc.frequency.setValueAtTime(880, now + 0.18);
        osc.frequency.exponentialRampToValueAtTime(440, now + 0.35);

        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);

        osc.start(now);
        osc.stop(now + 0.35);
      } else {
        // Crisp chime for QA Pass
        osc.type = 'sine';
        osc.frequency.setValueAtTime(587.33, now); // D5
        osc.frequency.setValueAtTime(880.00, now + 0.08); // A5

        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);

        osc.start(now);
        osc.stop(now + 0.25);
      }
    } catch (e) {
      console.warn('Audio feedback error:', e);
    }
  },

  updateVerdict(data) {
    const isAnomaly = data.verdict === 'CRITICAL_DEFECT';
    this.playAlert(isAnomaly);

    const verdictCard = document.getElementById('verdictCard');
    const verdictPill = document.getElementById('verdictPill');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictDesc = document.getElementById('verdictDesc');
    const routingBox = document.getElementById('routingBox');
    const routingActionText = document.getElementById('routingActionText');
    const unitIdDisplay = document.getElementById('unitIdDisplay');

    if (unitIdDisplay) unitIdDisplay.innerText = `UNIT: ${data.unit_id}`;

    if (verdictCard && verdictPill && verdictTitle && verdictDesc && routingBox && routingActionText) {
      if (isAnomaly) {
        verdictCard.className = 'glass-panel p-5 border-l-4 border-l-rose-500 glow-rose transition duration-300';
        verdictPill.className = 'text-xs font-extrabold px-2.5 py-0.5 rounded-full bg-rose-950 text-rose-400 border border-rose-700 font-mono';
        verdictPill.innerText = 'DEFECT DETECTED';
        verdictTitle.innerText = 'CRITICAL ANOMALY FLAGGED';
        verdictTitle.className = 'text-xl font-black tracking-tight text-rose-400 mb-1.5';
        verdictDesc.innerText = `Anomaly distance (${data.score}) exceeds decision threshold (${data.threshold}). Nearest-neighbor patches diverge from nominal baseline.`;
        
        routingBox.className = 'p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/60 flex items-center gap-3';
        routingActionText.innerText = data.conveyor_action;
        routingActionText.className = 'text-xs font-bold text-rose-300 tracking-wide font-mono';
      } else {
        verdictCard.className = 'glass-panel p-5 border-l-4 border-l-emerald-500 glow-emerald transition duration-300';
        verdictPill.className = 'text-xs font-extrabold px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-700 font-mono';
        verdictPill.innerText = 'QA PASSED';
        verdictTitle.innerText = 'QUALITY ASSURANCE PASSED';
        verdictTitle.className = 'text-xl font-black tracking-tight text-emerald-400 mb-1.5';
        verdictDesc.innerText = `Product appearance conforms strictly with pristine distribution (${data.score} < ${data.threshold}). Solder, traces, and surfaces are nominal.`;
        
        routingBox.className = 'p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-800/60 flex items-center gap-3';
        routingActionText.innerText = data.conveyor_action;
        routingActionText.className = 'text-xs font-bold text-emerald-300 tracking-wide font-mono';
      }
    }
  }
};

window.TelemetryManager = TelemetryManager;
