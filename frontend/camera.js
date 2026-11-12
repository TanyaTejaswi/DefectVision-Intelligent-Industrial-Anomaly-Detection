/**
 * DEFECTVISION 2.0 - HARDWARE CAMERA & OPTICAL SENSOR MODULE
 * Captures live frames from laptop webcam or GigE industrial USB cameras.
 */

const CameraController = {
  stream: null,
  videoElement: null,
  canvasElement: null,

  init(videoId = 'cameraVideo', canvasId = 'cameraCanvas') {
    this.videoElement = document.getElementById(videoId);
    this.canvasElement = document.getElementById(canvasId);
  },

  async startStream() {
    if (!this.videoElement) this.init();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Camera access is not supported by your current browser.');
      return false;
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment'
        }
      });
      this.videoElement.srcObject = this.stream;
      return true;
    } catch (err) {
      console.error('Camera stream initiation error:', err);
      alert('Unable to access video optical sensor: ' + err.message);
      return false;
    }
  },

  stopStream() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }
  },

  async captureFrameBlob() {
    if (!this.videoElement || !this.canvasElement) return null;

    const v = this.videoElement;
    const c = this.canvasElement;

    c.width = v.videoWidth || 900;
    c.height = v.videoHeight || 900;

    const ctx = c.getContext('2d');
    ctx.drawImage(v, 0, 0, c.width, c.height);

    return new Promise((resolve) => {
      c.toBlob((blob) => {
        if (!blob) {
          resolve(null);
          return;
        }
        const file = new File([blob], `aoi_optical_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
        resolve(file);
      }, 'image/jpeg', 0.95);
    });
  }
};

window.CameraController = CameraController;
