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
    // Currently highlighted saved annotation (null or {class, instance_id})
    highlightedAnnotation: null,

    // Class management
    classes: ['object'],
    selectedClass: 'object',
    instanceId: 1,
    classColors: {},       // className → [r, g, b]

    // Prompt
    promptType: 'points',  // 'points' | 'boxes' | 'text' | 'mixed' | 'points_boxes'
    textPrompt: '',

    // Point prompts
    points: [],            // [{x, y, label}]  label: 1=fg, 0=bg

    // Box drawing
    drawingBox: false,
    boxStart: null,        // {x, y} in canvas-pixel (natural image) coords
    boxEnd: null,
    boxes: [],             // finalized [[x1,y1,x2,y2], ...]
    boxLabels: [],         // parallel [1|0, ...]

    // Prompt insertion order — used by undoLastPrompt in points_boxes mode
    promptStack: [],       // 'point' | 'box'

    // Options
    filterByPrompt: false,

    // Debounce / busy flag
    isProcessing: false,
    debounceTimer: null,

    // Polarity of the box/point currently being drawn (1 = positive, 0 = negative)
    boxPolarity: 1,
};

// ── Module-level vars ──────────────────────────────────────────────────────
let imageCanvas, overlayCanvas, imageCtx, overlayCtx;

// Used in points_boxes mode: tracks mouse-down position to distinguish
// a stationary click (→ point) from a drag (→ box).
let _pxDown = null;

function $id(id) { return document.getElementById(id); }

// ── Bootstrap ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    imageCanvas   = $id('image-canvas');
    overlayCanvas = $id('overlay-canvas');
    imageCtx   = imageCanvas.getContext('2d');
    overlayCtx = overlayCanvas.getContext('2d');

    // Canvas pointer events live on the overlay (which sits on top)
    overlayCanvas.addEventListener('mousedown',   onMouseDown);
    overlayCanvas.addEventListener('mousemove',   onMouseMove);
    overlayCanvas.addEventListener('mouseup',     onMouseUp);
    overlayCanvas.addEventListener('mouseleave',  onMouseLeave);
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
                (_me.promptType === 'text' || _me.promptType === 'mixed') ? 'block' : 'none';
        });
    });

    $id('undo-btn').addEventListener('click', undoLastPrompt);
    $id('clear-instance-btn').addEventListener('click', clearInstancePrompts);
    $id('clear-all-btn').addEventListener('click', clearAllPrompts);
    $id('segment-btn').addEventListener('click', () => _meRunSAM());
    $id('save-btn').addEventListener('click', saveAnnotation);
    $id('filter-by-prompt').addEventListener('change', e => {
        _me.filterByPrompt = e.target.checked;
    });

    // Re-fit canvas when the window is resized
    window.addEventListener('resize', fitCanvasToContainer);

    loadClasses();
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
        const li   = document.createElement('li');
        li.title   = imgPath;
        li.dataset.path = imgPath;

        const thumb = document.createElement('img');
        thumb.className = 'thumb';
        thumb.src       = `/api/thumbnail/${imgPath}`;
        thumb.loading   = 'lazy';
        thumb.alt       = imgPath.split('/').pop();

        const name = document.createElement('span');
        name.className   = 'thumb-name';
        name.textContent = imgPath.split('/').pop();

        li.appendChild(thumb);
        li.appendChild(name);
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
    _me.currentImagePath      = relPath;
    _me.currentMasks          = [];
    _me.selectedMaskIdx       = 0;
    _me.boxes                 = [];
    _me.boxLabels             = [];
    _me.points                = [];
    _me.promptStack           = [];
    _me.annotations           = [];
    _me.highlightedAnnotation = null;

    clearOverlay();
    $id('mask-prev-container').innerHTML = '';
    $id('annotation-list').innerHTML = '';

    await new Promise((resolve, reject) => {
        const img = new window.Image();
        img.onload = () => {
            // Set canvas pixel dimensions to the image's natural resolution
            imageCanvas.width    = img.naturalWidth;
            imageCanvas.height   = img.naturalHeight;
            overlayCanvas.width  = img.naturalWidth;
            overlayCanvas.height = img.naturalHeight;
            imageCtx.drawImage(img, 0, 0);
            // Scale the CSS display size to fill the available area
            fitCanvasToContainer();
            resolve();
        };
        img.onerror = reject;
        img.src = `/api/image/${relPath}`;
    });

    setStatus(`${relPath}`);
    await loadAnnotations();
}

// ── Canvas fitting ─────────────────────────────────────────────────────────
function fitCanvasToContainer() {
    if (!imageCanvas.width || !imageCanvas.height) return;
    const area   = document.querySelector('.canvas-scroll');
    const availW = area.clientWidth;
    const availH = area.clientHeight;
    const scale  = Math.min(availW / imageCanvas.width, availH / imageCanvas.height);
    imageCanvas.style.width  = Math.floor(imageCanvas.width  * scale) + 'px';
    imageCanvas.style.height = Math.floor(imageCanvas.height * scale) + 'px';
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

async function loadClasses() {
    try {
        const resp = await fetch('/api/classes');
        const data = await resp.json();
        _me.classes = data.classes || ['object'];
        _me.selectedClass = _me.classes[0];
    } catch {
        _me.classes = ['object'];
        _me.selectedClass = 'object';
    }
    populateClassSelect();
}

async function addClass() {
    const input = $id('new-class-input');
    const name = input.value.trim();
    if (!name || _me.classes.includes(name)) return;

    try {
        const resp = await fetch('/api/classes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        if (!resp.ok) {
            setStatus('Failed to save class');
            return;
        }
        const data = await resp.json();
        _me.classes = data.classes;
    } catch {
        // Persist failed but still add locally so the session stays usable
        _me.classes.push(name);
    }

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
function canvasCoords(e) {
    const rect = overlayCanvas.getBoundingClientRect();
    return {
        x: Math.round((e.clientX - rect.left) * (overlayCanvas.width  / rect.width)),
        y: Math.round((e.clientY - rect.top)  * (overlayCanvas.height / rect.height)),
    };
}

// ── Highlighted annotation helper ──────────────────────────────────────────
function _clearHighlightedAnnotation() {
    if (!_me.highlightedAnnotation) return;
    _me.highlightedAnnotation = null;
    _me.currentMasks    = [];
    _me.selectedMaskIdx = 0;
    renderAnnotationList();
}

// ── Mouse event dispatch ───────────────────────────────────────────────────
function onMouseDown(e) {
    if (_me.promptType === 'points') {
        onPointMouseDown(e);
        return;
    }

    if (_me.promptType === 'points_boxes') {
        if (e.button !== 0 && e.button !== 2) return;
        e.preventDefault();
        _me.boxPolarity = e.button === 2 ? 0 : 1;
        _pxDown = canvasCoords(e);
        _me.boxStart = _pxDown;
        _me.boxEnd   = _pxDown;
        _me.drawingBox = false;
        return;
    }

    // Box / mixed modes — left-drag = positive, right-drag = negative
    if (e.button === 0) {
        _me.boxPolarity = 1;
    } else if (e.button === 2) {
        _me.boxPolarity = 0;
    } else {
        return;
    }
    _me.drawingBox = true;
    _me.boxStart   = canvasCoords(e);
    _me.boxEnd     = _me.boxStart;
}

function onMouseMove(e) {
    if (_me.promptType === 'points') return;

    if (_me.promptType === 'points_boxes') {
        if (!_pxDown) return;
        _me.boxEnd = canvasCoords(e);
        const dx = _me.boxEnd.x - _pxDown.x;
        const dy = _me.boxEnd.y - _pxDown.y;
        // Promote to drag once the cursor moves more than 5 px
        if (!_me.drawingBox && (Math.abs(dx) > 5 || Math.abs(dy) > 5)) {
            _me.drawingBox = true;
        }
        if (_me.drawingBox) {
            redrawOverlay();
            _drawBoxPreview(_pxDown, _me.boxEnd, _me.boxPolarity);
        }
        return;
    }

    if (!_me.drawingBox) return;
    _me.boxEnd = canvasCoords(e);
    redrawOverlay();
    _drawBoxPreview(_me.boxStart, _me.boxEnd, _me.boxPolarity);
}

function onMouseUp(e) {
    if (_me.promptType === 'points') return;

    if (_me.promptType === 'points_boxes') {
        if (!_pxDown) return;
        const coords = canvasCoords(e);
        if (_me.drawingBox) {
            const { x: x1, y: y1 } = _pxDown;
            const { x: x2, y: y2 } = coords;
            if (Math.abs(x2 - x1) > 3 && Math.abs(y2 - y1) > 3) {
                _me.boxes.push([Math.min(x1,x2), Math.min(y1,y2), Math.max(x1,x2), Math.max(y1,y2)]);
                _me.boxLabels.push(_me.boxPolarity);
                _me.promptStack.push('box');
                _clearHighlightedAnnotation();
                scheduleSegment();
            }
            _me.drawingBox = false;
        } else {
            // Stationary click → add a point
            _me.points.push({ ...coords, label: _me.boxPolarity });
            _me.promptStack.push('point');
            _clearHighlightedAnnotation();
            scheduleSegment();
        }
        _pxDown = null;
        redrawOverlay();
        return;
    }

    if (!_me.drawingBox) return;
    _me.drawingBox = false;
    const { x: x1, y: y1 } = _me.boxStart;
    const { x: x2, y: y2 } = _me.boxEnd;
    if (Math.abs(x2 - x1) > 3 && Math.abs(y2 - y1) > 3) {
        _me.boxes.push([Math.min(x1,x2), Math.min(y1,y2), Math.max(x1,x2), Math.max(y1,y2)]);
        _me.boxLabels.push(_me.boxPolarity);
        _me.promptStack.push('box');
        _clearHighlightedAnnotation();
        scheduleSegment();
    }
    redrawOverlay();
}

function onMouseLeave() {
    if (_me.drawingBox) {
        _me.drawingBox = false;
        _pxDown = null;
        redrawOverlay();
    } else if (_pxDown) {
        _pxDown = null;
    }
}

// ── Point handling ─────────────────────────────────────────────────────────
function onPointMouseDown(e) {
    if (e.button === 0) {
        _me.points.push({ ...canvasCoords(e), label: 1 });
        _me.promptStack.push('point');
    } else if (e.button === 2) {
        _me.points.push({ ...canvasCoords(e), label: 0 });
        _me.promptStack.push('point');
    } else {
        return;
    }
    _clearHighlightedAnnotation();
    redrawOverlay();
    scheduleSegment();
}

// ── Box preview helper ─────────────────────────────────────────────────────
function _drawBoxPreview(start, end, polarity) {
    overlayCtx.save();
    overlayCtx.strokeStyle = polarity === 0 ? 'rgba(220,50,50,0.9)' : 'rgba(255,255,255,0.85)';
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([6, 3]);
    overlayCtx.strokeRect(
        Math.min(start.x, end.x), Math.min(start.y, end.y),
        Math.abs(end.x - start.x), Math.abs(end.y - start.y),
    );
    overlayCtx.restore();
}

// ── Prompt management ──────────────────────────────────────────────────────

/**
 * Undo the last added prompt (last point or last box, based on insertion order).
 * In single-type modes (points/boxes) falls back to the relevant array.
 */
function undoLastPrompt() {
    const type = _me.promptType;
    let removed = false;

    if (type === 'points_boxes') {
        const last = _me.promptStack.pop();
        if (last === 'point' && _me.points.length > 0) {
            _me.points.pop();
            removed = true;
        } else if (last === 'box' && _me.boxes.length > 0) {
            _me.boxes.pop();
            _me.boxLabels.pop();
            removed = true;
        }
    } else if (type === 'points') {
        if (_me.points.length > 0) {
            _me.points.pop();
            _me.promptStack.pop();
            removed = true;
        }
    } else {
        // boxes / mixed / text
        if (_me.boxes.length > 0) {
            _me.boxes.pop();
            _me.boxLabels.pop();
            _me.promptStack.pop();
            removed = true;
        }
    }

    if (!removed) {
        setStatus('Nothing to undo');
        return;
    }

    const hasPrompts = _me.points.length > 0 || _me.boxes.length > 0;
    if (!hasPrompts) {
        _me.currentMasks    = [];
        _me.selectedMaskIdx = 0;
        clearOverlay();
        $id('mask-prev-container').innerHTML = '';
        setStatus('All prompts cleared');
    } else {
        scheduleSegment();
        setStatus('Last prompt removed');
    }
}

/**
 * Clear all prompts for the currently active mode — no confirmation required.
 *
 * - points      → clear _me.points
 * - boxes/mixed → clear _me.boxes + _me.boxLabels (+ text for mixed)
 * - text        → clear _me.textPrompt
 * - points_boxes→ clear both points and boxes
 */
function clearInstancePrompts() {
    const type = _me.promptType;
    let cleared = false;

    if (type === 'points') {
        if (_me.points.length > 0) {
            _me.points = [];
            _me.promptStack = _me.promptStack.filter(t => t !== 'point');
            cleared = true;
        }
    } else if (type === 'boxes') {
        if (_me.boxes.length > 0) {
            _me.boxes = [];
            _me.boxLabels = [];
            _me.promptStack = _me.promptStack.filter(t => t !== 'box');
            cleared = true;
        }
    } else if (type === 'points_boxes') {
        if (_me.points.length > 0 || _me.boxes.length > 0) {
            _me.points     = [];
            _me.boxes      = [];
            _me.boxLabels  = [];
            _me.promptStack = [];
            cleared = true;
        }
    } else if (type === 'text') {
        if (_me.textPrompt) {
            _me.textPrompt = '';
            $id('text-prompt-input').value = '';
            cleared = true;
        }
    } else if (type === 'mixed') {
        if (_me.boxes.length > 0 || _me.textPrompt) {
            _me.boxes      = [];
            _me.boxLabels  = [];
            _me.promptStack = [];
            _me.textPrompt = '';
            $id('text-prompt-input').value = '';
            cleared = true;
        }
    }

    if (!cleared) {
        setStatus('No prompts to clear');
        return;
    }

    _me.currentMasks    = [];
    _me.selectedMaskIdx = 0;
    clearOverlay();
    $id('mask-prev-container').innerHTML = '';
    setStatus('Instance prompts cleared');
}

/**
 * Clear ALL prompts (points, boxes, and text) across all modes.
 * Asks for confirmation before proceeding.
 */
function clearAllPrompts() {
    if (!window.confirm(
        'Clear ALL prompts (points, boxes, and text)?\nThis cannot be undone.'
    )) {
        return;
    }

    _me.points      = [];
    _me.boxes       = [];
    _me.boxLabels   = [];
    _me.textPrompt  = '';
    _me.promptStack = [];
    $id('text-prompt-input').value = '';

    _me.currentMasks    = [];
    _me.selectedMaskIdx = 0;
    clearOverlay();
    $id('mask-prev-container').innerHTML = '';
    setStatus('All prompts cleared');
}

function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// ── Overlay rendering ──────────────────────────────────────────────────────
function redrawOverlay() {
    clearOverlay();

    // 1. Render all returned masks simultaneously, then mark the selected one
    if (_me.currentMasks.length > 0) {
        renderAllMasks();
        drawSelectedMaskMarker();
    }

    // 2. Draw prompts on top
    const type = _me.promptType;
    if (type === 'points') {
        drawPoints();
    } else if (type === 'points_boxes') {
        drawPoints();   // both types visible simultaneously
        drawBoxes();
    } else {
        drawBoxes();
    }
}

function renderAllMasks() {
    if (_me.currentMasks.length === 0) return;
    const H = _me.currentMasks[0].length;
    const W = H > 0 ? _me.currentMasks[0][0].length : 0;
    if (!H || !W) return;

    const imgData = overlayCtx.createImageData(W, H);
    const buf     = imgData.data;

    for (let mi = 0; mi < _me.currentMasks.length; mi++) {
        const mask       = _me.currentMasks[mi];
        const isSelected = mi === _me.selectedMaskIdx;
        const [r, g, b]  = PALETTE[mi % PALETTE.length];
        const alpha      = isSelected ? 140 : 70;

        for (let row = 0; row < H; row++) {
            const maskRow = mask[row];
            const rowBase = row * W * 4;
            for (let col = 0; col < W; col++) {
                if (maskRow[col]) {
                    const i    = rowBase + col * 4;
                    buf[i]     = r;
                    buf[i + 1] = g;
                    buf[i + 2] = b;
                    buf[i + 3] = alpha;
                }
            }
        }
    }
    overlayCtx.putImageData(imgData, 0, 0);
}

// Draw an X at the bounding-box centre of the selected mask.
function drawSelectedMaskMarker() {
    if (_me.currentMasks.length === 0) return;
    const mask = _me.currentMasks[_me.selectedMaskIdx];
    if (!mask || mask.length === 0) return;

    let minRow = Infinity, maxRow = -Infinity;
    let minCol = Infinity, maxCol = -Infinity;
    for (let row = 0; row < mask.length; row++) {
        const maskRow = mask[row];
        for (let col = 0; col < maskRow.length; col++) {
            if (maskRow[col]) {
                if (row < minRow) minRow = row;
                if (row > maxRow) maxRow = row;
                if (col < minCol) minCol = col;
                if (col > maxCol) maxCol = col;
            }
        }
    }
    if (minRow === Infinity) return;

    const cx   = (minCol + maxCol) / 2;
    const cy   = (minRow + maxRow) / 2;
    const size = Math.max(18, Math.min(overlayCanvas.width, overlayCanvas.height) * 0.025);
    const lw   = Math.max(3, size * 0.22);

    const [r, g, b] = PALETTE[_me.selectedMaskIdx % PALETTE.length];

    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(255,255,255,0.85)';
    overlayCtx.lineWidth   = lw + 3;
    overlayCtx.lineCap     = 'round';
    overlayCtx.setLineDash([]);
    overlayCtx.beginPath();
    overlayCtx.moveTo(cx - size, cy - size);
    overlayCtx.lineTo(cx + size, cy + size);
    overlayCtx.moveTo(cx + size, cy - size);
    overlayCtx.lineTo(cx - size, cy + size);
    overlayCtx.stroke();
    overlayCtx.strokeStyle = `rgb(${r},${g},${b})`;
    overlayCtx.lineWidth   = lw;
    overlayCtx.beginPath();
    overlayCtx.moveTo(cx - size, cy - size);
    overlayCtx.lineTo(cx + size, cy + size);
    overlayCtx.moveTo(cx + size, cy - size);
    overlayCtx.lineTo(cx - size, cy + size);
    overlayCtx.stroke();
    overlayCtx.restore();
}

function drawPoints() {
    if (_me.points.length === 0) return;
    overlayCtx.save();
    const radius = Math.max(6, Math.round(Math.min(overlayCanvas.width, overlayCanvas.height) * 0.012));
    for (const pt of _me.points) {
        overlayCtx.beginPath();
        overlayCtx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
        overlayCtx.fillStyle   = pt.label === 1 ? '#22dd55' : '#dd2222';
        overlayCtx.fill();
        overlayCtx.strokeStyle = '#fff';
        overlayCtx.lineWidth   = Math.max(2, radius * 0.35);
        overlayCtx.stroke();
    }
    overlayCtx.restore();
}

function drawBoxes() {
    if (_me.boxes.length === 0) return;
    const [r, g, b] = classColor(_me.selectedClass);
    overlayCtx.save();
    overlayCtx.lineWidth = 2;
    for (let i = 0; i < _me.boxes.length; i++) {
        const [x1, y1, x2, y2] = _me.boxes[i];
        const isNeg = _me.boxLabels[i] === 0;
        overlayCtx.strokeStyle = isNeg ? 'rgb(220,50,50)' : `rgb(${r},${g},${b})`;
        overlayCtx.setLineDash(isNeg ? [6, 3] : []);
        overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    overlayCtx.restore();
}

// ── Multi-mask selector ────────────────────────────────────────────────────
function updateMaskPreviews(count) {
    const container = $id('mask-prev-container');
    container.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const [r, g, b] = PALETTE[i % PALETTE.length];

        const swatch = document.createElement('span');
        swatch.style.cssText = `
            display:inline-block;width:11px;height:11px;border-radius:50%;
            background:rgb(${r},${g},${b});margin-right:6px;vertical-align:middle;
            flex-shrink:0;
        `;

        const btn = document.createElement('button');
        btn.className = `mask-btn${i === 0 ? ' active' : ''}`;
        btn.appendChild(swatch);
        btn.appendChild(document.createTextNode(`Mask ${i + 1}`));
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

    if (promptType === 'points' && _me.points.length === 0) {
        setStatus('Click to place a point (right-click for background)');
        return;
    }
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
    if (promptType === 'points_boxes' && _me.points.length === 0 && _me.boxes.length === 0) {
        setStatus('Click (point) or drag (box) to add prompts');
        return;
    }

    _me.isProcessing = true;
    setStatus('Segmenting…');

    // Encode the image currently drawn on imageCanvas as JPEG base64
    const imageB64 = imageCanvas.toDataURL('image/jpeg', 0.92).split(',')[1];

    let promptData = {};
    if (promptType === 'points') {
        promptData = {
            points: _me.points.map(p => [p.x, p.y]),
            labels: _me.points.map(p => p.label),
        };
    } else if (promptType === 'text') {
        promptData = { text: _me.textPrompt };
    } else if (promptType === 'boxes') {
        promptData = { boxes: _me.boxes, labels: _me.boxLabels };
    } else if (promptType === 'mixed') {
        promptData = { text: _me.textPrompt, boxes: _me.boxes, labels: _me.boxLabels };
    } else {  // points_boxes
        promptData = {
            points:       _me.points.map(p => [p.x, p.y]),
            point_labels: _me.points.map(p => p.label),
            boxes:        _me.boxes,
            box_labels:   _me.boxLabels,
        };
    }

    try {
        const resp = await fetch('/api/segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_b64:        imageB64,
                prompt_type:      promptType,
                prompt_data:      promptData,
                filter_by_prompt: _me.filterByPrompt,
            }),
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
        _me.currentMasks    = (data.masks || []).map(m => decodeMaskRLE(m.shape, m.rle));
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

    const promptContent = {
        text:   _me.textPrompt,
        boxes:  _me.boxes,
        points: _me.points,
    };

    const startId  = _me.instanceId;
    const maskCount = _me.currentMasks.length;

    try {
        for (let i = 0; i < maskCount; i++) {
            const resp = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_path:     _me.currentImagePath,
                    class_name:     _me.selectedClass,
                    instance_id:    startId + i,
                    mask_data:      _me.currentMasks[i],
                    prompt_type:    _me.promptType,
                    prompt_content: promptContent,
                }),
            });
            if (!resp.ok) {
                setStatus(`Save failed for mask ${i + 1}`);
                return;
            }
        }
        _me.instanceId += maskCount;
        $id('instance-id-input').value = _me.instanceId;
        await loadAnnotations();
        setStatus(`${maskCount} annotation(s) saved`);
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
        const isHighlighted = _me.highlightedAnnotation &&
            _me.highlightedAnnotation.class === ann.class &&
            _me.highlightedAnnotation.instance_id === ann.instance_id;
        const li = document.createElement('li');
        li.className = isHighlighted ? 'ann-item ann-item--active' : 'ann-item';
        li.innerHTML = `
            <span class="ann-badge" style="background:rgb(${r},${g},${b});cursor:pointer" title="Click to highlight mask">
                ${escHtml(ann.class)} #${ann.instance_id}
            </span>
            <button class="delete-btn" title="Delete annotation">✕</button>
        `;
        li.querySelector('.ann-badge').addEventListener('click', () => selectAnnotation(ann));
        li.querySelector('.delete-btn').addEventListener('click', () => deleteAnnotation(ann));
        list.appendChild(li);
    }
}

async function selectAnnotation(ann) {
    // If already highlighted, deselect and clear overlay
    if (_me.highlightedAnnotation &&
        _me.highlightedAnnotation.class === ann.class &&
        _me.highlightedAnnotation.instance_id === ann.instance_id) {
        _me.highlightedAnnotation = null;
        _me.currentMasks    = [];
        _me.selectedMaskIdx = 0;
        clearOverlay();
        $id('mask-prev-container').innerHTML = '';
        renderAnnotationList();
        setStatus('Annotation deselected');
        return;
    }

    if (_me.classes.includes(ann.class)) {
        $id('class-select').value = ann.class;
        _me.selectedClass = ann.class;
    }
    _me.instanceId = ann.instance_id;
    $id('instance-id-input').value = ann.instance_id;

    // Clear active prompts so the canvas shows only the saved mask
    _me.boxes        = [];
    _me.boxLabels    = [];
    _me.points       = [];
    _me.promptStack  = [];
    $id('mask-prev-container').innerHTML = '';

    setStatus('Loading mask…');
    try {
        const params = new URLSearchParams({ class_name: ann.class, instance_id: ann.instance_id });
        const resp = await fetch(`/api/annotation/mask/${_me.currentImagePath}?${params}`);
        if (!resp.ok) {
            setStatus('Failed to load mask');
            return;
        }
        const data = await resp.json();
        _me.currentMasks          = [decodeMaskRLE(data.mask.shape, data.mask.rle)];
        _me.selectedMaskIdx       = 0;
        _me.highlightedAnnotation = { class: ann.class, instance_id: ann.instance_id };
        renderAnnotationList();
        redrawOverlay();
        setStatus(`Showing mask: ${ann.class} #${ann.instance_id}`);
    } catch (err) {
        setStatus('Network error loading mask');
        console.error('[mask_editor] mask load error:', err);
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

// ── RLE decode ─────────────────────────────────────────────────────────────
// Matches the server-side _rle_encode() format: alternating False/True run
// lengths starting with the False count (leading 0 when mask begins True).
function decodeMaskRLE(shape, rle) {
    const [H, W] = shape;
    const mask = new Array(H);
    for (let r = 0; r < H; r++) mask[r] = new Array(W).fill(false);
    let offset = 0;
    for (let i = 0; i < rle.length; i++) {
        const count = rle[i];
        if (i % 2 === 1) {
            for (let j = 0; j < count; j++) {
                mask[Math.floor((offset + j) / W)][(offset + j) % W] = true;
            }
        }
        offset += count;
    }
    return mask;
}

// ── Utility ────────────────────────────────────────────────────────────────
function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
