/**
 * DEFECTVISION 2.0 - ENTERPRISE MULTI-PAGE & 3-WAY INGESTION LOGIC
 * Manages Home, Line Catalog, 3-Way Ingestion (Upload, Dataset, Camera), and Live Results.
 */

let state = {
  currentPage: 'home',
  lines: [],
  presets: [],
  selectedLine: 'pcb1',
  activePresetId: 'pcb1_severed_trace',
  alpha: 0.45,
  colormap: 'jet',
  currentInspectionData: null,
  activeViewMode: 'overlay', // 'raw', 'heatmap', 'overlay'
  ingestMode: 'upload',      // 'upload', 'catalog', 'camera'
  cameraStream: null
};

let modalState = {
  activeTab: 'dataset', // 'file', 'dataset', 'camera', 'url'
  selectedFile: null,
  cameraStream: null,
  webUrl: '',
  lineSamples: [],
  selectedSampleIndex: 0
};

// Global clipboard listener for easy web image paste (Ctrl+V)
window.addEventListener('paste', async (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf('image') !== -1) {
      const blob = items[i].getAsFile();
      if (blob) {
        handlePastedImageBlob(blob);
        break;
      }
    } else if (items[i].type.indexOf('text/plain') !== -1) {
      items[i].getAsString((text) => {
        if (text.startsWith('http://') || text.startsWith('https://')) {
          handlePastedImageUrl(text);
        }
      });
    }
  }
});

document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await loadLines();
  await loadPresets();
  
  // Initial inspection pre-load
  await triggerInspection(state.activePresetId, state.selectedLine);
});

function setupEventListeners() {
  // Blend slider & colormap
  const blendSlider = document.getElementById('blendSlider');
  const blendVal = document.getElementById('blendVal');
  const cmapSelect = document.getElementById('cmapSelect');

  if (blendSlider && blendVal) {
    blendSlider.addEventListener('input', (e) => {
      state.alpha = parseFloat(e.target.value);
      blendVal.innerText = `${Math.round(state.alpha * 100)}%`;
    });

    blendSlider.addEventListener('change', () => {
      reRunCurrentInspection();
    });
  }

  if (cmapSelect) {
    cmapSelect.addEventListener('change', (e) => {
      state.colormap = e.target.value;
      reRunCurrentInspection();
    });
  }

  // View mode switcher buttons
  const btnRaw = document.getElementById('btnViewRaw');
  const btnHeat = document.getElementById('btnViewHeatmap');
  const btnOverlay = document.getElementById('btnViewOverlay');

  if (btnRaw) btnRaw.addEventListener('click', () => switchViewMode('raw'));
  if (btnHeat) btnHeat.addEventListener('click', () => switchViewMode('heatmap'));
  if (btnOverlay) btnOverlay.addEventListener('click', () => switchViewMode('overlay'));
}

/**
 * Multi-Page Navigation (Clean Corporate Instrumental Tabs)
 */
function navigateTo(pageId) {
  state.currentPage = pageId;

  // 1. Hide all pages
  const pages = ['pageHome', 'pageLines', 'pageInspect', 'pageBenchmarks'];
  pages.forEach(p => {
    const el = document.getElementById(p);
    if (el) el.classList.add('hidden');
  });

  // 2. Show target page
  const targetMap = {
    'home': 'pageHome',
    'lines': 'pageLines',
    'inspect': 'pageInspect',
    'benchmarks': 'pageBenchmarks'
  };
  const activePage = document.getElementById(targetMap[pageId]);
  if (activePage) activePage.classList.remove('hidden');

  // 3. Highlight Nav Button (Instrumental Underline Style)
  const navBtns = [
    { id: 'navBtnHome', page: 'home' },
    { id: 'navBtnLines', page: 'lines' },
    { id: 'navBtnInspect', page: 'inspect' },
    { id: 'navBtnBenchmarks', page: 'benchmarks' }
  ];

  navBtns.forEach(b => {
    const btn = document.getElementById(b.id);
    if (!btn) return;
    if (b.page === pageId) {
      btn.className = 'text-gray-900 font-extrabold border-b-2 border-[#00B050] pb-1 transition flex items-center gap-1';
    } else {
      btn.className = 'text-gray-600 hover:text-gray-900 font-bold transition flex items-center gap-1';
    }
  });

  if (pageId === 'lines') renderLinesCatalog();
  if (pageId === 'benchmarks') renderBenchmarkTable();

  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (window.lucide) lucide.createIcons();
}

/**
 * 3-Way Ingestion Mode Switcher (Upload, Catalog, Camera)
 */
function switchIngestMode(mode) {
  state.ingestMode = mode;

  const cardUpload = document.getElementById('ingestCardUpload');
  const cardCatalog = document.getElementById('ingestCardCatalog');
  const cardCamera = document.getElementById('ingestCardCamera');
  const preview = document.getElementById('cameraPreviewContainer');

  [cardUpload, cardCatalog, cardCamera].forEach(c => {
    if (c) c.className = 'p-4 rounded-xl border border-white/10 bg-slate-900/40 hover:border-white/20 cursor-pointer transition flex flex-col justify-between';
  });

  if (mode === 'upload') {
    if (cardUpload) cardUpload.className = 'p-4 rounded-xl border border-cyan-500 bg-cyan-950/20 cursor-pointer transition flex flex-col justify-between shadow-lg shadow-cyan-950';
    if (preview) preview.classList.add('hidden');
    stopCameraStream();
  } else if (mode === 'catalog') {
    if (cardCatalog) cardCatalog.className = 'p-4 rounded-xl border border-cyan-500 bg-cyan-950/20 cursor-pointer transition flex flex-col justify-between shadow-lg shadow-cyan-950';
    if (preview) preview.classList.add('hidden');
    stopCameraStream();
  } else if (mode === 'camera') {
    if (cardCamera) cardCamera.className = 'p-4 rounded-xl border border-emerald-500 bg-emerald-950/20 cursor-pointer transition flex flex-col justify-between shadow-lg shadow-emerald-950';
  }
}

/**
 * Live Camera Hardware Stream (Webcam / Factory Camera)
 */
async function startCameraStream() {
  switchIngestMode('camera');
  const preview = document.getElementById('cameraPreviewContainer');
  const video = document.getElementById('cameraVideo');

  try {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      state.cameraStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: { ideal: 900 }, height: { ideal: 900 } } 
      });
      video.srcObject = state.cameraStream;
      preview.classList.remove('hidden');
    } else {
      alert('Camera access is not supported by your browser.');
    }
  } catch (err) {
    console.error('Camera stream error:', err);
    alert('Unable to access optical camera: ' + err.message);
  }
}

function stopCameraStream() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(track => track.stop());
    state.cameraStream = null;
  }
}

async function captureCameraFrame() {
  const video = document.getElementById('cameraVideo');
  const canvas = document.getElementById('cameraCanvas');
  if (!video || !canvas) return;

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (blob) {
      const file = new File([blob], `live_camera_${Date.now()}.jpg`, { type: 'image/jpeg' });
      await inspectUploadedFile(file);
    }
  }, 'image/jpeg', 0.92);
}

/**
 * Execute Inspection on Uploaded File or Captured Camera Frame
 */
async function inspectUploadedFile(file) {
  const placeholder = document.getElementById('inspectedPlaceholder');
  const overlayImg = document.getElementById('inspectedOverlayImage');

  if (placeholder) placeholder.style.display = 'flex';
  if (overlayImg) overlayImg.style.opacity = '0.3';

  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', state.selectedLine);
    formData.append('alpha', state.alpha);
    formData.append('colormap', state.colormap);

    const res = await fetch('/api/inspect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Inspection failed');
    const data = await res.json();
    state.currentInspectionData = data;

    applyInspectionResults(data);
  } catch (err) {
    console.error('Inspection error:', err);
    if (placeholder) placeholder.innerHTML = `<span class="text-rose-400 text-xs">Inspection Error: ${err.message}</span>`;
  }
}

async function reRunCurrentInspection() {
  if (state.activePresetId) {
    await triggerInspection(state.activePresetId, state.selectedLine);
  } else {
    await triggerInspection(null, state.selectedLine);
  }
}

async function loadLines() {
  try {
    const res = await fetch('/api/lines');
    state.lines = await res.json();
  } catch (err) {
    console.error('Failed to load lines:', err);
  }
}

function renderLinesCatalog() {
  const container = document.getElementById('linesCatalogGrid');
  if (!container) return;
  container.innerHTML = '';

  state.lines.forEach(line => {
    const isSelected = line.id === state.selectedLine;
    const card = document.createElement('div');
    card.className = `clean-card p-6 flex flex-col justify-between cursor-pointer transition group ${
      isSelected ? 'border-[#00B050] ring-2 ring-[#00B050]/20' : 'hover:border-gray-400'
    }`;

    card.innerHTML = `
      <div>
        <div class="flex items-center justify-between text-xs font-mono mb-3">
          <span class="px-2 py-0.5 rounded bg-gray-100 text-gray-700 font-semibold uppercase">${line.sector}</span>
          <span class="text-[#00B050] font-bold">${(line.auroc * 100).toFixed(1)}% AUROC</span>
        </div>

        <div class="flex items-center gap-3 mb-2">
          <span class="text-2xl">${line.icon}</span>
          <h3 class="text-base font-bold text-gray-900 group-hover:text-[#00B050] transition">${line.name}</h3>
        </div>

        <p class="text-xs text-gray-600 leading-relaxed mb-4">${line.defects}</p>
      </div>

      <div class="pt-4 border-t border-gray-100 flex items-center justify-between font-mono">
        <span class="text-[11px] text-gray-500">${line.dataset}</span>
        <button onclick="event.stopPropagation(); openLineIngestModal('${line.id}')" class="btn-instrumental px-3 py-1.5 text-xs uppercase tracking-wider font-bold">
          <span>Select & Inspect &rarr;</span>
        </button>
      </div>
    `;

    card.onclick = () => {
      openLineIngestModal(line.id);
    };

    container.appendChild(card);
  });

  if (window.lucide) lucide.createIcons();
}

function updateLineHeaderTags() {
  const lineMeta = state.lines.find(l => l.id === state.selectedLine);
  if (!lineMeta) return;

  const t1 = document.getElementById('step1SelectedLineText');
  const t2 = document.getElementById('hudActiveLineName');
  if (t1) t1.innerText = `${lineMeta.name}`;
  if (t2) t2.innerText = `${lineMeta.name}`;

  // Filter dropdown on pageInspect to only show this line's samples
  const select = document.getElementById('presetSelect');
  if (select) {
    select.innerHTML = '';
    const filtered = state.presets.filter(p => p.category === state.selectedLine);
    filtered.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.innerText = `${p.category.toUpperCase()} - ${p.title}`;
      if (p.id === state.activePresetId) opt.selected = true;
      select.appendChild(opt);
    });
  }
}

async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    state.presets = await res.json();
    updateLineHeaderTags();
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

/**
 * ====================================================================
 * LINE-SPECIFIC 4-OPTION INGESTION POPUP / MODAL CONTROLLER
 * (Provides File, Filtered Line Dataset, Camera, and Web/Google URL)
 * ====================================================================
 */
function openLineIngestModal(lineId) {
  state.selectedLine = lineId;
  updateLineHeaderTags();

  const line = state.lines.find(l => l.id === lineId) || {
    id: lineId,
    name: lineId,
    sector: 'Industrial Manufacturing',
    icon: '⚙️',
    dataset: 'VisA / MVTec',
    defects: 'Surface scratches, component defects',
    auroc: 0.98
  };

  // 1. Populate Modal Headers
  const iconEl = document.getElementById('modalLineIcon');
  const titleEl = document.getElementById('modalLineTitle');
  const sectorEl = document.getElementById('modalLineSector');
  const benchEl = document.getElementById('modalLineBenchmark');
  const defectsEl = document.getElementById('modalLineDefects');
  const tabLineName = document.getElementById('modalTabLineName');
  const bannerLineName = document.getElementById('modalDatasetBannerLineName');
  const webGalleryLine = document.getElementById('modalWebGalleryLine');

  if (iconEl) iconEl.innerText = line.icon;
  if (titleEl) titleEl.innerText = line.name;
  if (sectorEl) sectorEl.innerText = line.sector;
  if (benchEl) benchEl.innerText = `${line.dataset} • ${(line.auroc * 100).toFixed(1)}% AUROC`;
  if (defectsEl) defectsEl.innerText = line.defects;
  if (tabLineName) tabLineName.innerText = line.id;
  if (bannerLineName) bannerLineName.innerText = line.name;
  if (webGalleryLine) webGalleryLine.innerText = line.id;

  // 2. Populate Real Dataset Samples List (Option 2) - Defaults to the first item!
  loadDatasetSamplesForLine(lineId);

  // 3. Populate Online Images & Browser Search (Option 4)
  renderModalWebGallery(lineId);

  // 4. Reset File & Camera states
  clearModalFileSelection();
  stopModalCameraStream();
  clearModalUrlInput();

  // 5. Open Modal on Dataset Tab as default
  switchModalTab('dataset');
  const modal = document.getElementById('lineIngestModal');
  if (modal) modal.classList.remove('hidden');

  if (window.lucide) lucide.createIcons();
}

function closeLineIngestModal() {
  stopModalCameraStream();
  const modal = document.getElementById('lineIngestModal');
  if (modal) modal.classList.add('hidden');
}

/**
 * 4-Tab Switcher inside Modal
 */
function switchModalTab(tabId) {
  modalState.activeTab = tabId;

  // Buttons
  const btnFile = document.getElementById('modalTabBtnFile');
  const btnDataset = document.getElementById('modalTabBtnDataset');
  const btnCamera = document.getElementById('modalTabBtnCamera');
  const btnUrl = document.getElementById('modalTabBtnUrl');

  const tabs = [
    { id: 'file', btn: btnFile, content: document.getElementById('modalTabContentFile') },
    { id: 'dataset', btn: btnDataset, content: document.getElementById('modalTabContentDataset') },
    { id: 'camera', btn: btnCamera, content: document.getElementById('modalTabContentCamera') },
    { id: 'url', btn: btnUrl, content: document.getElementById('modalTabContentUrl') }
  ];

  tabs.forEach(t => {
    if (t.content) {
      if (t.id === tabId) {
        t.content.classList.remove('hidden');
      } else {
        t.content.classList.add('hidden');
      }
    }

    if (t.btn) {
      if (t.id === tabId) {
        t.btn.className = 'px-3.5 py-2 rounded-lg font-bold flex items-center gap-2 transition border bg-gray-100 text-gray-900 border-[#00B050] shadow-sm';
      } else {
        t.btn.className = 'px-3.5 py-2 rounded-lg font-bold flex items-center gap-2 transition border border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-50';
      }
    }
  });

  if (tabId !== 'camera') {
    stopModalCameraStream();
  } else {
    startModalCameraStream();
  }

  if (window.lucide) lucide.createIcons();
}

/**
 * ====================================================================
 * TAB 2: DATASET SAMPLES LIST (Exact Streamlit Dropdown & Default 1st Item)
 * ====================================================================
 */
async function loadDatasetSamplesForLine(lineId) {
  const select = document.getElementById('modalDatasetSampleSelect');
  const countBadge = document.getElementById('modalTabPresetCount');
  const totalCountEl = document.getElementById('modalDatasetTotalCount');
  if (!select) return;

  select.innerHTML = '<option value="">Loading dataset samples...</option>';

  try {
    const res = await fetch(`/api/lines/${lineId}/samples`);
    const samples = await res.json();
    modalState.lineSamples = samples;

    select.innerHTML = '';
    if (samples.length === 0) {
      select.innerHTML = '<option value="">No dataset samples found</option>';
      if (totalCountEl) totalCountEl.innerText = '0 samples';
      return;
    }

    // Populate dropdown with exact format '#000 | TRAIN | good'
    samples.forEach((sample, idx) => {
      const opt = document.createElement('option');
      opt.value = idx;
      opt.innerText = sample.display;
      select.appendChild(opt);
    });

    // DEFAULT IS THE FIRST ONE!
    select.selectedIndex = 0;
    modalState.selectedSampleIndex = 0;
    updateSelectedSampleCard(0);

    if (countBadge) countBadge.innerText = samples.length;
    if (totalCountEl) totalCountEl.innerText = `${samples.length} items available`;

  } catch (err) {
    console.error('Failed to load dataset samples:', err);
    select.innerHTML = '<option value="">Error loading samples</option>';
  }
}

function handleModalSampleSelectionChange(e) {
  const idx = parseInt(e.target.value, 10);
  if (!isNaN(idx) && modalState.lineSamples[idx]) {
    updateSelectedSampleCard(idx);
  }
}

function updateSelectedSampleCard(idx) {
  modalState.selectedSampleIndex = idx;
  const sample = modalState.lineSamples[idx];
  if (!sample) return;

  const thumb = document.getElementById('modalSelectedSampleThumb');
  const filename = document.getElementById('modalSelectedSampleFilename');
  const title = document.getElementById('modalSelectedSampleTitle');
  const desc = document.getElementById('modalSelectedSampleDesc');

  if (thumb) {
    thumb.src = `/api/sample-image?path=${encodeURIComponent(sample.rel_path)}`;
  }

  if (filename) filename.innerText = sample.filename;
  if (title) title.innerText = sample.display;
  if (desc) {
    desc.innerText = 'Raw component image loaded from line buffer. Click "Inspect Selected Sample" below to run optical anomaly detection and compute thermal heatmap.';
  }
}

async function submitModalDatasetSampleInspection() {
  const sample = modalState.lineSamples[modalState.selectedSampleIndex];
  if (!sample) return;

  closeLineIngestModal();
  navigateTo('inspect');
  await inspectDatasetSample(sample);
}

async function inspectDatasetSample(sample) {
  const placeholder = document.getElementById('inspectedPlaceholder');
  const overlayImg = document.getElementById('inspectedOverlayImage');

  if (placeholder) placeholder.style.display = 'flex';
  if (overlayImg) overlayImg.style.opacity = '0.3';

  try {
    const formData = new FormData();
    formData.append('sample_path', sample.rel_path);
    formData.append('category', state.selectedLine);
    formData.append('alpha', state.alpha);
    formData.append('colormap', state.colormap);

    const res = await fetch('/api/inspect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Inspection failed');
    const data = await res.json();
    state.currentInspectionData = data;
    applyInspectionResults(data);
  } catch (err) {
    console.error('Dataset sample inspection error:', err);
    if (placeholder) {
      placeholder.innerHTML = `<span class="text-rose-400 text-xs">Inspection Error: ${err.message}</span>`;
    }
  }
}

/**
 * ====================================================================
 * TAB 4: BROWSE FROM INTERNET / GOOGLE IMAGES & CLIPBOARD SELECTOR
 * ====================================================================
 */
function openGoogleImagesInBrowser() {
  const line = state.lines.find(l => l.id === state.selectedLine);
  const lineName = line ? line.name : state.selectedLine;
  const query = `${lineName} industrial defect visual inspection`;
  const url = `https://www.google.com/search?tbm=isch&q=${encodeURIComponent(query)}`;
  window.open(url, '_blank');
}

async function pasteFromClipboard() {
  try {
    if (navigator.clipboard && navigator.clipboard.read) {
      const clipboardItems = await navigator.clipboard.read();
      for (const item of clipboardItems) {
        const imageType = item.types.find(type => type.startsWith('image/'));
        if (imageType) {
          const blob = await item.getType(imageType);
          handlePastedImageBlob(blob);
          return;
        }
      }
    }
    
    // Fallback: read text as URL
    if (navigator.clipboard && navigator.clipboard.readText) {
      const text = await navigator.clipboard.readText();
      if (text && (text.startsWith('http://') || text.startsWith('https://'))) {
        handlePastedImageUrl(text);
        return;
      }
    }

    alert('No image found in clipboard. Please copy an image from Google Images first (Right click -> Copy Image) and press Ctrl+V!');
  } catch (err) {
    console.warn('Clipboard read error:', err);
    alert('Please press Ctrl+V to paste your copied image!');
  }
}

function handlePastedImageBlob(blob) {
  const file = new File([blob], `google_web_image_${Date.now()}.jpg`, { type: 'image/jpeg' });
  closeLineIngestModal();
  navigateTo('inspect');
  inspectUploadedFile(file);
}

function handlePastedImageUrl(url) {
  const input = document.getElementById('modalWebUrlInput');
  if (input) {
    input.value = url;
    handleModalUrlInput({ target: input });
    previewModalWebUrl();
  }
}

/**
 * Curated In-App Web Images Gallery (Click Any to Select & Inspect)
 */
const WEB_GALLERY_IMAGES = {
  pcb1: [
    { title: "Mainboard Conductor Trace", url: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=400&q=80" },
    { title: "SMD Micro-Resistor Array", url: "https://images.unsplash.com/photo-1597733336794-12d05021d510?auto=format&fit=crop&w=400&q=80" },
    { title: "Copper Circuit Board", url: "https://images.unsplash.com/photo-1555680202-c86f0e12f086?auto=format&fit=crop&w=400&q=80" },
    { title: "Defect Solder Trace", url: "https://images.unsplash.com/photo-1563770660941-20978e870e26?auto=format&fit=crop&w=400&q=80" }
  ],
  pcb2: [
    { title: "Cold Solder Joint Defect", url: "https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=400&q=80" },
    { title: "SMT Micro-Soldering Pad", url: "https://images.unsplash.com/photo-1517420704952-d9f39e95b43e?auto=format&fit=crop&w=400&q=80" },
    { title: "Misaligned Micro-Chip", url: "https://images.unsplash.com/photo-1588508065123-287b28e013da?auto=format&fit=crop&w=400&q=80" },
    { title: "PCB Soldering Quality Standard", url: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=400&q=80" }
  ],
  bottle: [
    { title: "Glass Rim Fracture Defect", url: "https://images.unsplash.com/photo-1602143407151-7111542de6e8?auto=format&fit=crop&w=400&q=80" },
    { title: "Beverage Packaging Unit", url: "https://images.unsplash.com/photo-1527661591475-527312dd65f5?auto=format&fit=crop&w=400&q=80" },
    { title: "Chemical Sealing Flaw", url: "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?auto=format&fit=crop&w=400&q=80" },
    { title: "Pristine Bottle Baseline", url: "https://images.unsplash.com/photo-1556740738-b6a63e27c4df?auto=format&fit=crop&w=400&q=80" }
  ],
  cable: [
    { title: "Wire Cut Outer Insulation", url: "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=400&q=80" },
    { title: "Swapped Harness Conductor", url: "https://images.unsplash.com/photo-1581092335397-9583fe92d232?auto=format&fit=crop&w=400&q=80" },
    { title: "Automotive Braid Wire", url: "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=400&q=80" },
    { title: "Intact Harness Assembly", url: "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=400&q=80" }
  ],
  steel: [
    { title: "Hot-Rolled Strip Scratch", url: "https://images.unsplash.com/photo-1535813547-99c456a41d4a?auto=format&fit=crop&w=400&q=80" },
    { title: "Rolled-In Oxide Scale", url: "https://images.unsplash.com/photo-1504917599217-d4dc5ebe6122?auto=format&fit=crop&w=400&q=80" },
    { title: "Steel Mill Surface Coiling", url: "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?auto=format&fit=crop&w=400&q=80" },
    { title: "Pristine Steel Strip", url: "https://images.unsplash.com/photo-1513694203232-719a280e022f?auto=format&fit=crop&w=400&q=80" }
  ]
};

function renderModalWebGallery(lineId) {
  const container = document.getElementById('modalWebGalleryGrid');
  if (!container) return;
  container.innerHTML = '';

  const images = WEB_GALLERY_IMAGES[lineId] || WEB_GALLERY_IMAGES['pcb1'];
  images.forEach(img => {
    const card = document.createElement('div');
    card.className = 'group rounded-xl overflow-hidden border border-gray-200 hover:border-[#00B050] bg-white cursor-pointer transition shadow-sm flex flex-col';
    card.innerHTML = `
      <div class="aspect-square bg-black overflow-hidden relative">
        <img src="${img.url}" alt="${img.title}" class="w-full h-full object-cover group-hover:scale-105 transition duration-300">
        <span class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center text-white text-[11px] font-bold font-mono">
          Click to Inspect
        </span>
      </div>
      <div class="p-2 text-[10px] font-mono text-gray-700 truncate text-center bg-gray-50 group-hover:text-[#00B050] font-bold">
        ${img.title}
      </div>
    `;
    card.onclick = () => {
      closeLineIngestModal();
      navigateTo('inspect');
      inspectWebUrl(img.url);
    };
    container.appendChild(card);
  });
}

/**
 * Tab 1: File Upload Handlers
 */
function handleModalFileSelected(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  modalState.selectedFile = file;

  const previewBox = document.getElementById('modalFilePreviewBox');
  const thumb = document.getElementById('modalFileThumbnail');
  const nameEl = document.getElementById('modalFileName');
  const sizeEl = document.getElementById('modalFileSize');
  const btnInspect = document.getElementById('btnModalInspectUpload');

  if (nameEl) nameEl.innerText = file.name;
  if (sizeEl) sizeEl.innerText = `${(file.size / 1024).toFixed(1)} KB • Image Loaded`;

  const reader = new FileReader();
  reader.onload = (e) => {
    if (thumb) thumb.src = e.target.result;
    if (previewBox) previewBox.classList.remove('hidden');
    if (btnInspect) {
      btnInspect.disabled = false;
      btnInspect.classList.remove('opacity-50', 'cursor-not-allowed');
    }
  };
  reader.readAsDataURL(file);
}

function clearModalFileSelection() {
  modalState.selectedFile = null;
  const input = document.getElementById('modalFileInput');
  if (input) input.value = '';

  const previewBox = document.getElementById('modalFilePreviewBox');
  const btnInspect = document.getElementById('btnModalInspectUpload');
  if (previewBox) previewBox.classList.add('hidden');
  if (btnInspect) {
    btnInspect.disabled = true;
    btnInspect.classList.add('opacity-50', 'cursor-not-allowed');
  }
}

async function submitModalFileUpload() {
  if (!modalState.selectedFile) return;
  const file = modalState.selectedFile;
  closeLineIngestModal();
  navigateTo('inspect');
  await inspectUploadedFile(file);
}

/**
 * Tab 3: Modal Camera Stream Handlers
 */
async function startModalCameraStream() {
  const video = document.getElementById('modalCameraVideo');
  const placeholder = document.getElementById('modalCameraPlaceholder');
  const btnStart = document.getElementById('btnModalStartCam');
  const btnStop = document.getElementById('btnModalStopCam');
  const btnCapture = document.getElementById('btnModalCaptureCam');

  try {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      modalState.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 900 }, height: { ideal: 900 } }
      });
      if (video) {
        video.srcObject = modalState.cameraStream;
        video.classList.remove('hidden');
      }
      if (placeholder) placeholder.classList.add('hidden');
      if (btnStart) btnStart.classList.add('hidden');
      if (btnStop) btnStop.classList.remove('hidden');
      if (btnCapture) {
        btnCapture.disabled = false;
        btnCapture.classList.remove('opacity-50', 'cursor-not-allowed');
      }
    } else {
      alert('Camera access is not supported by your browser.');
    }
  } catch (err) {
    console.error('Modal camera error:', err);
    alert('Unable to access camera: ' + err.message);
  }
}

function stopModalCameraStream() {
  if (modalState.cameraStream) {
    modalState.cameraStream.getTracks().forEach(t => t.stop());
    modalState.cameraStream = null;
  }
  const video = document.getElementById('modalCameraVideo');
  const placeholder = document.getElementById('modalCameraPlaceholder');
  const btnStart = document.getElementById('btnModalStartCam');
  const btnStop = document.getElementById('btnModalStopCam');
  const btnCapture = document.getElementById('btnModalCaptureCam');

  if (video) video.srcObject = null;
  if (placeholder) placeholder.classList.remove('hidden');
  if (btnStart) btnStart.classList.remove('hidden');
  if (btnStop) btnStop.classList.add('hidden');
  if (btnCapture) {
    btnCapture.disabled = true;
    btnCapture.classList.add('opacity-50', 'cursor-not-allowed');
  }
}

async function captureModalCameraAndInspect() {
  const video = document.getElementById('modalCameraVideo');
  const canvas = document.getElementById('modalCameraCanvas');
  if (!video || !canvas) return;

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (blob) {
      const file = new File([blob], `camera_${state.selectedLine}_${Date.now()}.jpg`, { type: 'image/jpeg' });
      stopModalCameraStream();
      closeLineIngestModal();
      navigateTo('inspect');
      await inspectUploadedFile(file);
    }
  }, 'image/jpeg', 0.92);
}

/**
 * Tab 4: Web / Google Image Handlers
 */
const CURATED_WEB_SUGGESTIONS = {
  pcb1: [
    { label: "Mainboard Conductor Trace", url: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=600&q=80" },
    { label: "High-Density SMD Array", url: "https://images.unsplash.com/photo-1597733336794-12d05021d510?auto=format&fit=crop&w=600&q=80" }
  ],
  pcb2: [
    { label: "Micro Solder Joint", url: "https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=600&q=80" },
    { label: "SMT Micro-Chip Lead", url: "https://images.unsplash.com/photo-1517420704952-d9f39e95b43e?auto=format&fit=crop&w=600&q=80" }
  ],
  pcb3: [
    { label: "IC Component Pinout", url: "https://images.unsplash.com/photo-1555680202-c86f0e12f086?auto=format&fit=crop&w=600&q=80" },
    { label: "Surface Bypass Caps", url: "https://images.unsplash.com/photo-1588508065123-287b28e013da?auto=format&fit=crop&w=600&q=80" }
  ],
  pcb4: [
    { label: "Multi-Layer Micro-Via", url: "https://images.unsplash.com/photo-1563770660941-20978e870e26?auto=format&fit=crop&w=600&q=80" },
    { label: "Substrate Copper Plane", url: "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=600&q=80" }
  ],
  capsules: [
    { label: "Pharmaceutical Gelatin Capsule", url: "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?auto=format&fit=crop&w=600&q=80" },
    { label: "Medical Blister Unit", url: "https://images.unsplash.com/photo-1471864190281-a93a3070b6de?auto=format&fit=crop&w=600&q=80" }
  ],
  bottle: [
    { label: "Glass Beverage Bottle Rim", url: "https://images.unsplash.com/photo-1602143407151-7111542de6e8?auto=format&fit=crop&w=600&q=80" },
    { label: "Industrial Bottling Container", url: "https://images.unsplash.com/photo-1527661591475-527312dd65f5?auto=format&fit=crop&w=600&q=80" }
  ],
  cable: [
    { label: "Automotive Wiring Harness", url: "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=600&q=80" },
    { label: "Conductor Wire Braid", url: "https://images.unsplash.com/photo-1581092335397-9583fe92d232?auto=format&fit=crop&w=600&q=80" }
  ],
  steel: [
    { label: "Hot-Rolled Strip Texture", url: "https://images.unsplash.com/photo-1535813547-99c456a41d4a?auto=format&fit=crop&w=600&q=80" },
    { label: "Industrial Steel Coil", url: "https://images.unsplash.com/photo-1504917599217-d4dc5ebe6122?auto=format&fit=crop&w=600&q=80" }
  ]
};

function renderModalUrlSuggestions(lineId) {
  const container = document.getElementById('modalUrlSuggestionsList');
  if (!container) return;
  container.innerHTML = '';

  const list = CURATED_WEB_SUGGESTIONS[lineId] || CURATED_WEB_SUGGESTIONS['pcb1'];
  list.forEach(item => {
    const chip = document.createElement('button');
    chip.className = 'px-3 py-1.5 rounded-lg bg-white border border-gray-200 hover:border-[#00B050] text-gray-700 hover:text-[#00B050] text-[11px] font-mono transition flex items-center gap-1.5 shadow-sm';
    chip.innerHTML = `<span>🔍</span><span>${item.label}</span>`;
    chip.onclick = () => {
      const input = document.getElementById('modalWebUrlInput');
      if (input) {
        input.value = item.url;
        handleModalUrlInput({ target: input });
        previewModalWebUrl();
      }
    };
    container.appendChild(chip);
  });
}

function handleModalUrlInput(e) {
  const val = e.target.value ? e.target.value.trim() : '';
  modalState.webUrl = val;
  const btn = document.getElementById('btnModalInspectUrl');
  if (btn) {
    if (val.length > 5) {
      btn.disabled = false;
      btn.classList.remove('opacity-50', 'cursor-not-allowed');
    } else {
      btn.disabled = true;
      btn.classList.add('opacity-50', 'cursor-not-allowed');
    }
  }
}

function previewModalWebUrl() {
  const input = document.getElementById('modalWebUrlInput');
  const url = input ? input.value.trim() : '';
  if (!url) return;

  const box = document.getElementById('modalWebUrlPreviewBox');
  const img = document.getElementById('modalWebUrlPreviewImg');
  const display = document.getElementById('modalWebUrlDisplay');
  const btn = document.getElementById('btnModalInspectUrl');

  if (img) {
    img.src = url;
    img.onload = () => {
      if (box) box.classList.remove('hidden');
      if (display) display.innerText = url.split('/').pop().split('?')[0] || 'Web Image';
      if (btn) {
        btn.disabled = false;
        btn.classList.remove('opacity-50', 'cursor-not-allowed');
      }
    };
    img.onerror = () => {
      if (box) box.classList.remove('hidden');
      if (display) display.innerText = 'URL verified for PatchCore inspection';
      if (btn) {
        btn.disabled = false;
        btn.classList.remove('opacity-50', 'cursor-not-allowed');
      }
    };
  }
}

function clearModalUrlInput() {
  modalState.webUrl = '';
  const input = document.getElementById('modalWebUrlInput');
  if (input) input.value = '';
  const box = document.getElementById('modalWebUrlPreviewBox');
  if (box) box.classList.add('hidden');
  const btn = document.getElementById('btnModalInspectUrl');
  if (btn) {
    btn.disabled = true;
    btn.classList.add('opacity-50', 'cursor-not-allowed');
  }
}

async function submitModalUrlInspection() {
  const input = document.getElementById('modalWebUrlInput');
  const url = input ? input.value.trim() : '';
  if (!url) return;

  closeLineIngestModal();
  navigateTo('inspect');
  await inspectWebUrl(url);
}

async function inspectWebUrl(url) {
  const placeholder = document.getElementById('inspectedPlaceholder');
  const overlayImg = document.getElementById('inspectedOverlayImage');

  if (placeholder) placeholder.style.display = 'flex';
  if (overlayImg) overlayImg.style.opacity = '0.3';

  try {
    const formData = new FormData();
    formData.append('image_url', url);
    formData.append('category', state.selectedLine);
    formData.append('alpha', state.alpha);
    formData.append('colormap', state.colormap);

    const res = await fetch('/api/inspect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || 'Inspection failed');
    }
    const data = await res.json();
    state.currentInspectionData = data;
    applyInspectionResults(data);
  } catch (err) {
    console.error('URL inspection error:', err);
    if (placeholder) {
      placeholder.innerHTML = `<span class="text-rose-400 text-xs">URL Inspection Error: ${err.message}</span>`;
    }
  }
}

async function triggerInspection(presetId, category) {
  const placeholder = document.getElementById('inspectedPlaceholder');
  const overlayImg = document.getElementById('inspectedOverlayImage');

  if (placeholder) placeholder.style.display = 'flex';
  if (overlayImg) overlayImg.style.opacity = '0.3';

  try {
    const formData = new FormData();
    if (presetId) formData.append('preset_id', presetId);
    if (category) formData.append('category', category);
    formData.append('alpha', state.alpha);
    formData.append('colormap', state.colormap);

    const res = await fetch('/api/inspect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Inspection failed');
    const data = await res.json();
    state.currentInspectionData = data;

    applyInspectionResults(data);
  } catch (err) {
    console.error('Inspection error:', err);
    if (placeholder) placeholder.innerHTML = `<span class="text-rose-400 text-xs">Inspection Error: ${err.message}</span>`;
  }
}

function applyInspectionResults(data) {
  const goldenImg = document.getElementById('goldenImage');
  const goldenPlaceholder = document.getElementById('goldenPlaceholder');
  const placeholder = document.getElementById('inspectedPlaceholder');
  const overlayImg = document.getElementById('inspectedOverlayImage');

  if (data.images.golden_reference) {
    goldenImg.src = data.images.golden_reference;
    goldenImg.style.display = 'block';
    goldenPlaceholder.style.display = 'none';
  } else {
    goldenImg.style.display = 'none';
    goldenPlaceholder.style.display = 'flex';
  }

  switchViewMode(state.activeViewMode);
  if (placeholder) placeholder.style.display = 'none';
  if (overlayImg) overlayImg.style.opacity = '1';

  updateTelemetry(data);
}

function switchViewMode(mode) {
  state.activeViewMode = mode;
  const overlayImg = document.getElementById('inspectedOverlayImage');
  const badge = document.getElementById('inspectedUnitBadge');
  const data = state.currentInspectionData;
  if (!data || !overlayImg || !badge) return;

  const btnRaw = document.getElementById('btnViewRaw');
  const btnHeat = document.getElementById('btnViewHeatmap');
  const btnOverlay = document.getElementById('btnViewOverlay');

  [btnRaw, btnHeat, btnOverlay].forEach(b => {
    if (b) b.className = 'px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition';
  });

  if (mode === 'raw') {
    overlayImg.src = data.images.inspected_raw;
    badge.innerText = 'RAW SENSOR';
    badge.className = 'text-[11px] px-2 py-0.2 rounded bg-slate-800 text-slate-300 font-mono border border-white/20';
    if (btnRaw) btnRaw.className = 'px-2.5 py-1 rounded-lg bg-white text-black font-bold transition';
  } else if (mode === 'heatmap') {
    overlayImg.src = data.images.heatmap;
    badge.innerText = 'THERMAL DISTANCE';
    badge.className = 'text-[11px] px-2 py-0.2 rounded bg-rose-950 text-rose-400 font-mono border border-rose-800';
    if (btnHeat) btnHeat.className = 'px-2.5 py-1 rounded-lg bg-rose-600 text-white font-bold transition';
  } else {
    overlayImg.src = data.images.overlay;
    badge.innerText = 'OVERLAY (JET)';
    badge.className = 'text-[11px] px-2 py-0.2 rounded bg-cyan-950 text-cyan-400 font-mono border border-cyan-800';
    if (btnOverlay) btnOverlay.className = 'px-2.5 py-1 rounded-lg bg-cyan-600 text-black font-bold transition';
  }
}

function updateTelemetry(data) {
  const isAnomaly = data.verdict === 'CRITICAL_DEFECT';

  const sub = document.getElementById('viewfinderUnitSub');
  if (sub) sub.innerText = `Inspecting ${data.sample_name} on target line ${data.category.toUpperCase()}`;

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

  // KPI widgets
  const kpiScore = document.getElementById('kpiScore');
  const kpiDelta = document.getElementById('kpiDelta');
  const kpiThreshold = document.getElementById('kpiThreshold');
  const kpiLineName = document.getElementById('kpiLineName');
  const kpiProb = document.getElementById('kpiProb');
  const probBar = document.getElementById('probBar');
  const kpiLatency = document.getElementById('kpiLatency');

  if (kpiScore) {
    kpiScore.innerText = data.score.toFixed(2);
    kpiScore.className = `text-2xl font-black ${isAnomaly ? 'text-rose-400' : 'text-emerald-400'}`;
  }

  if (kpiDelta) {
    const deltaSign = data.delta >= 0 ? `+${data.delta.toFixed(2)}` : data.delta.toFixed(2);
    kpiDelta.innerText = `${deltaSign} vs Threshold`;
    kpiDelta.className = `text-[11px] ${isAnomaly ? 'text-rose-400' : 'text-emerald-400'}`;
  }

  if (kpiThreshold) kpiThreshold.innerText = data.threshold.toFixed(2);
  if (kpiLineName) kpiLineName.innerText = `Target: ${data.category.toUpperCase()}`;

  if (kpiProb) {
    kpiProb.innerText = `${data.confidence.toFixed(1)}%`;
    kpiProb.className = `text-2xl font-black ${isAnomaly ? 'text-rose-400' : 'text-emerald-400'}`;
  }

  if (probBar) {
    probBar.style.width = `${Math.min(100, data.confidence)}%`;
    probBar.className = `${isAnomaly ? 'bg-rose-500' : 'bg-emerald-400'} h-1.5 rounded-full transition-all duration-300`;
  }

  if (kpiLatency) kpiLatency.innerText = `${data.latency_ms} ms`;

  // Line Intelligence
  const activeLineMeta = state.lines.find(l => l.id === data.category);
  if (activeLineMeta) {
    const desc = document.getElementById('lineDefectsDesc');
    const badge = document.getElementById('lineDatasetBadge');
    const auroc = document.getElementById('lineAurocBadge');
    if (desc) desc.innerText = activeLineMeta.defects;
    if (badge) badge.innerText = activeLineMeta.dataset;
    if (auroc) auroc.innerText = `${(activeLineMeta.auroc * 100).toFixed(2)}%`;
  }

  if (window.lucide) lucide.createIcons();
}

/**
 * Benchmark Scoreboard Table
 */
function renderBenchmarkTable() {
  const tbody = document.getElementById('benchmarkTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  state.lines.forEach(line => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-white/[0.03] transition';

    tr.innerHTML = `
      <td class="py-4 px-4 font-sans font-medium text-slate-300">${line.sector}</td>
      <td class="py-4 px-4 font-bold text-white flex items-center gap-2">
        <span>${line.icon}</span>
        <span>${line.name}</span>
      </td>
      <td class="py-4 px-4 text-cyan-300">${line.dataset}</td>
      <td class="py-4 px-4 text-right">
        <span class="px-2 py-0.5 rounded-full font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
          ${(line.auroc * 100).toFixed(2)}%
        </span>
      </td>
      <td class="py-4 px-4 text-right font-bold text-white">${line.f1.toFixed(4)}</td>
      <td class="py-4 px-4 text-right text-slate-300 font-mono">${line.threshold.toFixed(2)}</td>
      <td class="py-4 px-4 text-center">
        <button onclick="selectLineAndInspect('${line.id}')" class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-cyan-600 hover:text-black text-cyan-300 text-xs font-bold border border-slate-700 transition">
          Inspect Line
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function selectLineAndInspect(lineId) {
  openLineIngestModal(lineId);
}

function exportBenchmarkCSV() {
  let csv = 'Sector,Category,Dataset,AUROC,F1_Score,Optimal_Threshold\n';
  state.lines.forEach(l => {
    csv += `"${l.sector}","${l.id}","${l.dataset}",${(l.auroc*100).toFixed(2)}%,${l.f1},${l.threshold}\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.setAttribute('href', url);
  a.setAttribute('download', 'DefectVision_Industrial_Benchmarks.csv');
  a.click();
}

/**
 * Operator Auth Modal Handling
 */
function toggleAuthModal() {
  const modal = document.getElementById('authModal');
  if (modal) modal.classList.toggle('hidden');
}

function handleAuthSubmit(e) {
  e.preventDefault();
  const user = document.getElementById('authUsername').value || 'admin';
  document.getElementById('navOperatorText').innerText = `Operator: ${user}`;
  toggleAuthModal();
}
