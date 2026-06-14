'use strict';

// ── Color palette (one hue per class, cycling) ─────────────────────────────
const PALETTE = [
    [255,  82,  82],
    [ 82, 194, 255],
    [ 82, 255, 130],
    [255, 200,  82],
    [200,  82, 255],
    [ 82, 255, 210],
    [255, 140,   0],
    [160, 255,   0],
];

// ── Application state ──────────────────────────────────────────────────────
const _me = {
    images: [],
    currentImagePath: null,

    // Returned masks from last SAM call (array of 2-D bool arrays)
    currentMasks: [],
    selectedMaskIdx: 0,

    annotations: [],

    // Class management
    classes: ['object'],
    selectedClass: 'object',
    instanceId: 1,
    classColors: {},       // className → [r, g, b]

    // Prompt
    promptType: 'boxes',   // 'boxes' | 'text' | 'mixed'
    textPrompt: '',

    // Box drawing
    drawingBox: false,
    boxStart: null,        // {x, y} in canvas-pixel (natural image) coords
    boxEnd: null,
    boxes: [],             // finalized [[x1,y1,x2,y2], ...]
    boxLabels: [],         // parallel [1, ...]

    // Debounce / busy flag
    isProcessing: false,
    debounceTimer: null,
};

// ── DOM refs ───────────────────────────────────────────────────────────────
let imageCanvas, overlayCanvas, imageCtx, overlayCtx;

function $id(id) { return document.getElementById(id); }

// ── Bootstrap ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    imageCanvas  = $id('image-canvas');
    overlayCanvas = $id('overlay-canvas');
    imageCtx  = imageCanvas.getContext('2d');
    overlayCtx = overlayCanvas.getContext('2d');

    // Canvas pointer events live on the overlay (which sits on top)
    overlayCanvas.addEventListener('mousedown', onMouseDown);
    overlayCanvas.addEventListener('mousemove', onMouseMove);
    overlayCanvas.addEventListener('mouseup',   onMouseUp);
    overlayCanvas.addEventListener('mouseleave', onMouseLeave);
    overlayCanvas.addEventListener('contextmenu', e => e.preventDefault());

    // Sidebar controls
    $id('add-class-btn').addEventListener('click', addClass);
    $id('new-class-input').addEventListener('keydown', e => {
        if (e.key === 'Enter') addClass();
    });
    $id('class-select').addEventListener('change', () => {
        _me.selectedClass = $id('class-select').value;
        updateInstanceId();
        redrawOverlay();
    });
    $id('instance-id-input').addEventListener('change', () => {
        _me.instanceId = Math.max(1, parseInt($id('instance-id-input').value, 10) || 1);
    });
    $id('text-prompt-input').addEventListener('input', e => {
        _me.textPrompt = e.target.value;
    });

    document.querySelectorAll('input[name="prompt-type"]').forEach(radio => {
        radio.addEventListener('change', () => {
            _me.promptType = radio.value;
            $id('text-prompt-section').style.display =
                (_me.promptType !== 'boxes') ? 'block' : 'none';
        });
    });

    $id('clear-boxes-btn').addEventListener('click', clearBoxes);
    $id('segment-btn').addEventListener('click', () => _meRunSAM());
    $id('save-btn').addEventListener('click', saveAnnotation);

    populateClassSelect();
    loadImageList();
});

function setStatus(msg) {
    $id('status-bar').textContent = msg;
}

// ── Gallery ────────────────────────────────────────────────────────────────
async function loadImageList() {
    try {
        const resp = await fetch('/api/images');
        const data = await resp.json();
        _me.images = data.images || [];
    } catch {
        setStatus('Failed to load image list');
        return;
    }
    const list = $id('image-list');
    list.innerHTML = '';
    for (const imgPath of _me.images) {
        const li = document.createElement('li');
        li.textContent = imgPath.split('/').pop();
        li.title = imgPath;
        li.dataset.path = imgPath;
        li.addEventListener('click', () => {
            document.querySelectorAll('#image-list li').forEach(el => el.classList.remove('active'));
            li.classList.add('active');
            loadImage(imgPath);
        });
        list.appendChild(li);
    }
    setStatus(`${_me.images.length} image(s) found`);
}

async function loadImage(relPath) {
    _me.currentImagePath = relPath;
    _me.currentMasks     = [];
    _me.selectedMaskIdx  = 0;
    _me.boxes            = [];
    _me.boxLabels        = [];
    _me.annotations      = [];

    clearOverlay();
    $id('mask-prev-container').innerHTML = '';
    $id('annotation-list').innerHTML = '';

    await new Promise((resolve, reject) => {
        const img = new window.Image();
        img.onload = () => {
            // Set canvas pixel dimensions to the image's natural resolution
            imageCanvas.width  = img.naturalWidth;
            imageCanvas.height = img.naturalHeight;
            overlayCanvas.width  = img.naturalWidth;
            overlayCanvas.height = img.naturalHeight;
            imageCtx.drawImage(img, 0, 0);
            resolve();
        };
        img.onerror = reject;
        img.src = `/api/image/${relPath}`;
    });

    setStatus(`${relPath}`);
    await loadAnnotations();
}

// ── Class management ───────────────────────────────────────────────────────
function classColor(name) {
    if (!_me.classColors[name]) {
        const idx = Object.keys(_me.classColors).length % PALETTE.length;
        _me.classColors[name] = PALETTE[idx];
    }
    return _me.classColors[name];
}

function populateClassSelect() {
    const sel = $id('class-select');
    sel.innerHTML = '';
    for (const cls of _me.classes) {
        const opt = document.createElement('option');
        opt.value = cls;
        opt.textContent = cls;
        sel.appendChild(opt);
    }
    sel.value = _me.selectedClass;
}

function addClass() {
    const input = $id('new-class-input');
    const name = input.value.trim();
    if (!name || _me.classes.includes(name)) return;
    _me.classes.push(name);
    populateClassSelect();
    $id('class-select').value = name;
    _me.selectedClass = name;
    input.value = '';
    updateInstanceId();
}

// ── Instance ID ────────────────────────────────────────────────────────────
function updateInstanceId() {
    const maxId = _me.annotations
        .filter(a => a.class === _me.selectedClass)
        .reduce((m, a) => Math.max(m, a.instance_id), 0);
    _me.instanceId = maxId + 1;
    $id('instance-id-input').value = _me.instanceId;
}

// ── Canvas coordinate conversion ───────────────────────────────────────────
// Returns coordinates in canvas-pixel space (= natural image resolution),
// accounting for any CSS scaling applied to the displayed canvas.
function canvasCoords(e) {
    const rect = overlayCanvas.getBoundingClientRect();
    return {
        x: Math.round((e.clientX - rect.left) * (overlayCanvas.width  / rect.width)),
        y: Math.round((e.clientY - rect.top)  * (overlayCanvas.height / rect.height)),
    };
}

// ── Box drawing ────────────────────────────────────────────────────────────
function onMouseDown(e) {
    if (e.button !== 0) return;
    const pt = canvasCoords(e);
    _me.drawingBox = true;
    _me.boxStart   = pt;
    _me.boxEnd     = pt;
}

function onMouseMove(e) {
    if (!_me.drawingBox) return;
    _me.boxEnd = canvasCoords(e);
    redrawOverlay();
    // Live dashed preview of the box being drawn
    const { x: x1, y: y1 } = _me.boxStart;
    const { x: x2, y: y2 } = _me.boxEnd;
    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(255,255,255,0.85)';
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([6, 3]);
    overlayCtx.strokeRect(
        Math.min(x1, x2), Math.min(y1, y2),
        Math.abs(x2 - x1), Math.abs(y2 - y1),
    );
    overlayCtx.restore();
}

function onMouseUp(e) {
    if (!_me.drawingBox) return;
    _me.drawingBox = false;
    const { x: x1, y: y1 } = _me.boxStart;
    const { x: x2, y: y2 } = _me.boxEnd;
    // Only register boxes large enough to be meaningful (> 3 px in each dim)
    if (Math.abs(x2 - x1) > 3 && Math.abs(y2 - y1) > 3) {
        _me.boxes.push([Math.min(x1,x2), Math.min(y1,y2), Math.max(x1,x2), Math.max(y1,y2)]);
        _me.boxLabels.push(1);
        scheduleSegment();
    }
    redrawOverlay();
}

function onMouseLeave() {
    if (_me.drawingBox) {
        _me.drawingBox = false;
        redrawOverlay();
    }
}

function clearBoxes() {
    _me.boxes        = [];
    _me.boxLabels    = [];
    _me.currentMasks = [];
    _me.selectedMaskIdx = 0;
    clearOverlay();
    $id('mask-prev-container').innerHTML = '';
    setStatus('Boxes cleared');
}

function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// ── Overlay rendering ──────────────────────────────────────────────────────
function redrawOverlay() {
    clearOverlay();

    // 1. Render the selected mask as a coloured RGBA overlay
    if (_me.currentMasks.length > 0 && _me.selectedMaskIdx < _me.currentMasks.length) {
        renderMaskOverlay(_me.currentMasks[_me.selectedMaskIdx]);
    }

    // 2. Draw all finalized boxes in the class colour
    drawBoxes();
}

function renderMaskOverlay(maskArray) {
    const H = maskArray.length;
    const W = H > 0 ? maskArray[0].length : 0;
    if (!H || !W) return;

    const [r, g, b] = classColor(_me.selectedClass);
    // Uint8ClampedArray: each pixel is 4 bytes (RGBA)
    const imgData = overlayCtx.createImageData(W, H);
    const buf     = imgData.data; // Uint8ClampedArray

    for (let row = 0; row < H; row++) {
        const maskRow = maskArray[row];
        const rowBase = row * W * 4;
        for (let col = 0; col < W; col++) {
            if (maskRow[col]) {
                const i     = rowBase + col * 4;
                buf[i]      = r;
                buf[i + 1]  = g;
                buf[i + 2]  = b;
                buf[i + 3]  = 102; // ~40% opacity
            }
        }
    }
    overlayCtx.putImageData(imgData, 0, 0);
}

function drawBoxes() {
    if (_me.boxes.length === 0) return;
    const [r, g, b] = classColor(_me.selectedClass);
    overlayCtx.save();
    overlayCtx.strokeStyle = `rgb(${r},${g},${b})`;
    overlayCtx.lineWidth   = 2;
    overlayCtx.setLineDash([]);
    for (const [x1, y1, x2, y2] of _me.boxes) {
        overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    overlayCtx.restore();
}

// ── Multi-mask selector ────────────────────────────────────────────────────
function updateMaskPreviews(count) {
    const container = $id('mask-prev-container');
    container.innerHTML = '';
    if (count <= 1) return;
    for (let i = 0; i < count; i++) {
        const btn = document.createElement('button');
        btn.className  = `mask-btn${i === 0 ? ' active' : ''}`;
        btn.textContent = `Mask ${i + 1}`;
        btn.addEventListener('click', () => {
            _me.selectedMaskIdx = i;
            container.querySelectorAll('.mask-btn').forEach((b, j) => {
                b.className = `mask-btn${j === i ? ' active' : ''}`;
            });
            redrawOverlay();
        });
        container.appendChild(btn);
    }
}

// ── SAM 3 inference ────────────────────────────────────────────────────────
function scheduleSegment() {
    if (_me.debounceTimer !== null) clearTimeout(_me.debounceTimer);
    _me.debounceTimer = setTimeout(_meRunSAM, 400);
}

async function _meRunSAM() {
    _me.debounceTimer = null;
    if (_me.isProcessing || !_me.currentImagePath) return;

    const promptType = _me.promptType;
    if (promptType === 'boxes' && _me.boxes.length === 0) {
        setStatus('Draw a bounding box first');
        return;
    }
    if (promptType === 'text' && !_me.textPrompt.trim()) {
        setStatus('Enter a text prompt first');
        return;
    }
    if (promptType === 'mixed' && _me.boxes.length === 0 && !_me.textPrompt.trim()) {
        setStatus('Provide a box or text prompt');
        return;
    }

    _me.isProcessing = true;
    setStatus('Segmenting…');

    // Encode the image currently drawn on imageCanvas as JPEG base64
    const imageB64 = imageCanvas.toDataURL('image/jpeg', 0.92).split(',')[1];

    let promptData = {};
    if (promptType === 'text') {
        promptData = { text: _me.textPrompt };
    } else if (promptType === 'boxes') {
        promptData = { boxes: _me.boxes, labels: _me.boxLabels };
    } else {
        promptData = { text: _me.textPrompt, boxes: _me.boxes, labels: _me.boxLabels };
    }

    try {
        const resp = await fetch('/api/segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: imageB64, prompt_type: promptType, prompt_data: promptData }),
        });

        if (resp.status === 429) {
            setStatus('Model busy — try again');
            return;
        }
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            setStatus(`Error ${resp.status}: ${err.error || 'unknown'}`);
            return;
        }

        const data = await resp.json();
        _me.currentMasks    = data.masks || [];
        _me.selectedMaskIdx = 0;
        redrawOverlay();
        updateMaskPreviews(data.count);
        setStatus(`${data.count} mask(s) returned`);
    } catch (err) {
        setStatus('Network error');
        console.error('[mask_editor] segment error:', err);
    } finally {
        _me.isProcessing = false;
    }
}

// ── Save annotation ────────────────────────────────────────────────────────
async function saveAnnotation() {
    if (!_me.currentImagePath) {
        setStatus('No image loaded');
        return;
    }
    if (_me.currentMasks.length === 0) {
        setStatus('No mask to save — segment first');
        return;
    }

    const maskData = _me.currentMasks[_me.selectedMaskIdx];
    const promptContent = { text: _me.textPrompt, boxes: _me.boxes };

    try {
        const resp = await fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path:    _me.currentImagePath,
                class_name:    _me.selectedClass,
                instance_id:   _me.instanceId,
                mask_data:     maskData,
                prompt_type:   _me.promptType,
                prompt_content: promptContent,
            }),
        });

        if (resp.ok) {
            _me.instanceId++;
            $id('instance-id-input').value = _me.instanceId;
            await loadAnnotations();
            setStatus('Annotation saved');
        } else {
            setStatus('Save failed');
        }
    } catch (err) {
        setStatus('Network error on save');
        console.error('[mask_editor] save error:', err);
    }
}

// ── Annotations ────────────────────────────────────────────────────────────
async function loadAnnotations() {
    if (!_me.currentImagePath) return;
    try {
        const resp = await fetch(`/api/annotations/${_me.currentImagePath}`);
        const data = await resp.json();
        _me.annotations = data.annotations || [];
    } catch {
        _me.annotations = [];
    }
    renderAnnotationList();
    updateInstanceId();
}

function renderAnnotationList() {
    const list = $id('annotation-list');
    list.innerHTML = '';
    for (const ann of _me.annotations) {
        const [r, g, b] = classColor(ann.class);
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="ann-badge" style="background:rgb(${r},${g},${b})">
                ${escHtml(ann.class)} #${ann.instance_id}
            </span>
            <button class="delete-btn" title="Delete annotation">✕</button>
        `;
        li.querySelector('.delete-btn').addEventListener('click', () => deleteAnnotation(ann));
        list.appendChild(li);
    }
}

async function deleteAnnotation(ann) {
    try {
        const resp = await fetch('/api/annotation', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path:  _me.currentImagePath,
                class_name:  ann.class,
                instance_id: ann.instance_id,
            }),
        });
        if (resp.ok) {
            await loadAnnotations();
            setStatus('Annotation deleted');
        }
    } catch (err) {
        console.error('[mask_editor] delete error:', err);
    }
}

// ── Utility ────────────────────────────────────────────────────────────────
function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
