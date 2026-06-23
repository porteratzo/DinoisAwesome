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
    // Cached decoded masks for all saved annotations: "class:instance_id" → 2-D bool array
    annotationMasks: {},

    // Class management
    classes: ['object'],
    selectedClass: 'object',
    instanceId: 1,
    classColors: {},       // className → [r, g, b]

    // Which SAM backend is active — set by loadModelInfo() from /api/info.
    // false = Sam3Model (concept model, text+boxes only, no native points)
    // true  = Sam3TrackerModel (interactive, native points, no negative boxes)
    useTracker: false,

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

    // Brush tool
    brushSize: 20,       // radius in image pixels
    isBrushing: false,   // true while mouse button held in brush mode
    brushErasing: false, // true = right-click erase stroke

    // Options
    filterByPrompt: false,

    // Debounce / busy flag
    isProcessing: false,
    debounceTimer: null,

    // Polarity of the box/point currently being drawn (1 = positive, 0 = negative)
    boxPolarity: 1,

    // ── Zoom / pan ─────────────────────────────────────────────────────────
    // CSS transform on .canvas-stack; canvasCoords() uses getBoundingClientRect()
    // so coordinate conversion stays correct at any zoom level.
    zoom: 1,
    panX: 0,
    panY: 0,
    isPanning: false,
    _panAnchorX: 0,
    _panAnchorY: 0,
    _panStartX: 0,
    _panStartY: 0,
};

// ── Module-level vars ──────────────────────────────────────────────────────
let imageCanvas, overlayCanvas, imageCtx, overlayCtx;

// Off-screen canvas that accumulates brush strokes (image-pixel sized).
// White-with-alpha; color is applied at render time via compositing.
let _brushCanvas = null;
let _brushCtx    = null;

// Used in points_boxes mode: tracks mouse-down position to distinguish
// a stationary click (→ point) from a drag (→ box).
let _pxDown = null;

function $id(id) { return document.getElementById(id); }

// ── Zoom / pan ─────────────────────────────────────────────────────────────

/**
 * Clamp panX/panY using standard image-viewer rules:
 *  - If the canvas is larger than the scroll container on an axis, clamp so
 *    the canvas edge never retreats past the container edge (no black border).
 *  - If the canvas is smaller, lock pan to 0 on that axis (keep it centered).
 */
function _clampPan() {
    const scrollEl = document.querySelector('.canvas-scroll');
    if (!scrollEl || !imageCanvas.offsetWidth) return;

    const scrollW = scrollEl.clientWidth;
    const scrollH = scrollEl.clientHeight;
    // Natural (pre-transform) CSS layout size of the canvas-stack
    const nW = imageCanvas.offsetWidth;
    const nH = imageCanvas.offsetHeight;

    // Half-extents of the canvas in screen pixels at current zoom
    const halfW = (nW / 2) * _me.zoom;
    const halfH = (nH / 2) * _me.zoom;
    const halfSW = scrollW / 2;
    const halfSH = scrollH / 2;

    // X axis
    if (halfW > halfSW) {
        // Canvas wider than container: clamp so neither edge shows a gap
        const limit = halfW - halfSW;
        _me.panX = Math.max(-limit, Math.min(limit, _me.panX));
    } else {
        _me.panX = 0;
    }

    // Y axis
    if (halfH > halfSH) {
        const limit = halfH - halfSH;
        _me.panY = Math.max(-limit, Math.min(limit, _me.panY));
    } else {
        _me.panY = 0;
    }
}

function _applyTransform() {
    _clampPan();
    const stack = document.querySelector('.canvas-stack');
    if (stack) {
        stack.style.transform = `translate(${_me.panX}px,${_me.panY}px) scale(${_me.zoom})`;
    }
    _updateCanvasInfo();
}

function _resetView() {
    _me.zoom = 1;
    _me.panX = 0;
    _me.panY = 0;
    _applyTransform();
}

/** Update the right-side info chip in the status bar. */
function _updateCanvasInfo() {
    const el = $id('canvas-info');
    if (!el) return;
    const parts = [];
    if (imageCanvas && imageCanvas.width && imageCanvas.height) {
        parts.push(`${imageCanvas.width}\u00d7${imageCanvas.height}`);
    }
    parts.push(`${Math.round(_me.zoom * 100)}%`);
    el.textContent = parts.join(' · ');
}

// ── Wheel: Ctrl+scroll → zoom, plain scroll → pan ─────────────────────────
function _onScrollWheel(e) {
    e.preventDefault();
    const scrollEl = document.querySelector('.canvas-scroll');
    const scrollRect = scrollEl.getBoundingClientRect();
    // Natural center of .canvas-stack (flex-centered, no transform)
    const cx0 = scrollRect.left + scrollRect.width  / 2;
    const cy0 = scrollRect.top  + scrollRect.height / 2;

    if (e.ctrlKey) {
        // Zoom toward cursor
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const s  = _me.zoom;
        const s1 = Math.max(0.1, Math.min(15, s * factor));
        if (s !== 0) {
            _me.panX = (e.clientX - cx0) * (1 - s1 / s) + _me.panX * (s1 / s);
            _me.panY = (e.clientY - cy0) * (1 - s1 / s) + _me.panY * (s1 / s);
        }
        _me.zoom = s1;
    } else {
        // Pan: vertical scroll moves Y, horizontal (trackpad / shift+scroll) moves X
        _me.panX -= e.deltaX;
        _me.panY -= e.deltaY;
    }
    _applyTransform();
}

// ── Middle-mouse-button drag → pan ─────────────────────────────────────────
function _onScrollMouseDown(e) {
    if (e.button !== 1) return;
    e.preventDefault();
    _me.isPanning    = true;
    _me._panAnchorX  = e.clientX;
    _me._panAnchorY  = e.clientY;
    _me._panStartX   = _me.panX;
    _me._panStartY   = _me.panY;
    const scrollEl = document.querySelector('.canvas-scroll');
    if (scrollEl) scrollEl.style.cursor = 'grabbing';
}

function _onScrollMouseMove(e) {
    if (!_me.isPanning) return;
    _me.panX = _me._panStartX + (e.clientX - _me._panAnchorX);
    _me.panY = _me._panStartY + (e.clientY - _me._panAnchorY);
    _applyTransform();
}

function _onScrollMouseUp() {
    if (!_me.isPanning) return;
    _me.isPanning = false;
    const scrollEl = document.querySelector('.canvas-scroll');
    if (scrollEl) scrollEl.style.cursor = '';
}

// ── Model info ─────────────────────────────────────────────────────────────
async function loadModelInfo() {
    try {
        const resp = await fetch('/api/info');
        const data = await resp.json();
        _me.useTracker = !!data.use_tracker;
        const el = $id('model-info');
        if (el) {
            const name    = (data.model_id || 'unknown').split('/').pop();
            const tracker = _me.useTracker ? ' · tracker' : '';
            el.textContent = `Model: ${name}${tracker}`;
        }
    } catch { /* non-critical — leave default text */ }
    _updatePromptAvailability();
}

/**
 * Grey out prompt-type radio buttons that are unsupported by the active model
 * and annotate them with a hover tooltip explaining the limitation and fix.
 *
 * Sam3Model (useTracker=false):
 *   - No native point support. Sam3Processor accepts only text and boxes.
 *     Points are approximated as tiny micro-boxes, giving unreliable results.
 *   → Disables: "points", "points_boxes"
 *
 * Sam3TrackerModel (useTracker=true):
 *   - No negative-box support. Sam3TrackerProcessor has no box-label argument;
 *     negative boxes are silently dropped.
 *   → Warns at draw-time (not disabled here — positive boxes still work).
 */
function _updatePromptAvailability() {
    // Per-mode limitation messages (null = available).
    const SAM3_WARNINGS = {
        points:
            'Not available with Sam3Model.\n\n' +
            'Sam3Processor has no native point input — clicks are converted to tiny ' +
            'bounding boxes internally, which gives unreliable results.\n\n' +
            'Fix: set USE_TRACKER=1 to load Sam3TrackerModel, which has native point support.',
        points_boxes:
            'Not available with Sam3Model.\n\n' +
            'The point half of this mode is unsupported (Sam3Processor has no native ' +
            'input_points argument). Only boxes would reach the model.\n\n' +
            'Fix: set USE_TRACKER=1, or use the "Boxes" mode directly.',
    };

    document.querySelectorAll('input[name="prompt-type"]').forEach(radio => {
        const label = radio.closest('label') || radio.parentElement;
        const warning = !_me.useTracker ? (SAM3_WARNINGS[radio.value] || null) : null;

        if (warning) {
            radio.disabled = true;
            label.classList.add('prompt-unavailable');
            label.title = warning;
            // If this mode is currently selected, fall back to boxes.
            if (_me.promptType === radio.value) {
                const fallback = document.querySelector('input[name="prompt-type"][value="boxes"]');
                if (fallback) {
                    fallback.checked = true;
                    _me.promptType = 'boxes';
                }
            }
        } else {
            radio.disabled = false;
            label.classList.remove('prompt-unavailable');
            label.title = '';
        }
    });
}

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
            _updateBrushCursorStyle();
            _updateBrushUI();
        });
    });

    $id('brush-size-slider').addEventListener('input', e => {
        _me.brushSize = parseInt(e.target.value, 10);
        $id('brush-size-value').textContent = _me.brushSize;
    });

    $id('init-from-sam-btn').addEventListener('click', initBrushFromSAMMask);
    $id('clear-brush-btn').addEventListener('click', clearBrushCanvas);

    $id('undo-btn').addEventListener('click', undoLastPrompt);
    $id('clear-instance-btn').addEventListener('click', clearInstancePrompts);
    $id('clear-all-btn').addEventListener('click', clearAllPrompts);
    $id('segment-btn').addEventListener('click', () => _meRunSAM());
    $id('save-btn').addEventListener('click', saveAnnotation);
    $id('filter-by-prompt').addEventListener('change', e => {
        _me.filterByPrompt = e.target.checked;
    });

    // Re-fit canvas when the window is resized (also resets view)
    window.addEventListener('resize', () => { fitCanvasToContainer(); _resetView(); });

    // Zoom / pan on the scroll container
    const scrollEl = document.querySelector('.canvas-scroll');
    scrollEl.addEventListener('wheel',     _onScrollWheel,     { passive: false });
    scrollEl.addEventListener('mousedown', _onScrollMouseDown);
    scrollEl.addEventListener('mousemove', _onScrollMouseMove);
    scrollEl.addEventListener('mouseup',   _onScrollMouseUp);
    scrollEl.addEventListener('mouseleave', _onScrollMouseUp);

    // Reset view button + Ctrl+0 keyboard shortcut
    $id('reset-view-btn').addEventListener('click', _resetView);
    window.addEventListener('keydown', e => {
        if (e.ctrlKey && e.key === '0') { e.preventDefault(); _resetView(); }
    });

    loadClasses();
    loadImageList();
    loadModelInfo();
});

function setStatus(msg) {
    // status-bar is now a <span> inside the .status-bar container
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

    _me.annotationMasks = {};
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
            _initBrushCanvas();
            // Scale the CSS display size to fill the available area, then reset view
            fitCanvasToContainer();
            _resetView();
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
    if (e.button === 1) return;  // middle-mouse handled by canvas-scroll pan handler

    if (_me.promptType === 'brush') {
        if (e.button !== 0 && e.button !== 2) return;
        e.preventDefault();
        _me.isBrushing  = true;
        _me.brushErasing = (e.button === 2);
        const coords = canvasCoords(e);
        _brushPaint(coords.x, coords.y, _me.brushErasing);
        redrawOverlay();
        _drawBrushCursor(coords.x, coords.y);
        return;
    }

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
        // Sam3TrackerModel has no negative-box API — warn and block right-drag.
        if (_me.useTracker) {
            setStatus(
                '⚠ Negative boxes are not supported by Sam3TrackerModel ' +
                '(Sam3TrackerProcessor has no box-label argument — they would be silently dropped). ' +
                'Use right-click point prompts in "Points" or "Points + Boxes" mode to exclude regions.'
            );
            return;
        }
    } else {
        return;
    }
    _me.drawingBox = true;
    _me.boxStart   = canvasCoords(e);
    _me.boxEnd     = _me.boxStart;
}

function onMouseMove(e) {
    if (_me.promptType === 'brush') {
        const coords = canvasCoords(e);
        if (_me.isBrushing) {
            _brushPaint(coords.x, coords.y, _me.brushErasing);
            redrawOverlay();
        } else {
            redrawOverlay();
        }
        _drawBrushCursor(coords.x, coords.y);
        return;
    }

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
    if (_me.promptType === 'brush') {
        _me.isBrushing = false;
        return;
    }

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
    if (_me.promptType === 'brush') {
        _me.isBrushing = false;
        redrawOverlay();  // erase the cursor circle
        return;
    }

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
    if (_me.promptType === 'brush') {
        setStatus('Undo is not supported for brush — use Clear Brush Mask');
        return;
    }
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
    if (_me.promptType === 'brush') {
        clearBrushCanvas();
        return;
    }
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

    // 1. Render all saved annotation masks as background layer
    renderAllAnnotationMasks();

    // 2. Render SAM-returned masks on top (if any are active)
    if (_me.currentMasks.length > 0) {
        renderAllMasks();
        drawSelectedMaskMarker();
    }

    // 3. Render brush mask (visible regardless of prompt mode so it persists)
    renderBrushMask();

    // 4. Draw prompts on top (not in brush mode — brush cursor is drawn separately)
    const type = _me.promptType;
    if (type === 'points') {
        drawPoints();
    } else if (type === 'points_boxes') {
        drawPoints();
        drawBoxes();
    } else if (type !== 'brush') {
        drawBoxes();
    }
}

function renderAllAnnotationMasks() {
    if (_me.annotations.length === 0) return;
    const keys = Object.keys(_me.annotationMasks);
    if (keys.length === 0) return;

    // Determine canvas dimensions from the first available mask
    const firstMask = _me.annotationMasks[keys[0]];
    const H = firstMask.length;
    const W = H > 0 ? firstMask[0].length : 0;
    if (!H || !W) return;

    const imgData = overlayCtx.createImageData(W, H);
    const buf     = imgData.data;

    for (const ann of _me.annotations) {
        const key  = `${ann.class}:${ann.instance_id}`;
        const mask = _me.annotationMasks[key];
        if (!mask) continue;
        const [r, g, b] = classColor(ann.class);
        const isHighlighted = _me.highlightedAnnotation &&
            _me.highlightedAnnotation.class === ann.class &&
            _me.highlightedAnnotation.instance_id === ann.instance_id;
        const alpha = isHighlighted ? 170 : 80;

        for (let row = 0; row < H; row++) {
            const maskRow = mask[row];
            const rowBase = row * W * 4;
            for (let col = 0; col < W; col++) {
                if (maskRow[col]) {
                    const i = rowBase + col * 4;
                    // Alpha-composite: new color over whatever is already in buf
                    const bgA   = buf[i + 3] / 255;
                    const fgA   = alpha / 255;
                    const outA  = fgA + bgA * (1 - fgA);
                    if (outA > 0) {
                        buf[i]     = Math.round((r * fgA + buf[i]     * bgA * (1 - fgA)) / outA);
                        buf[i + 1] = Math.round((g * fgA + buf[i + 1] * bgA * (1 - fgA)) / outA);
                        buf[i + 2] = Math.round((b * fgA + buf[i + 2] * bgA * (1 - fgA)) / outA);
                        buf[i + 3] = Math.round(outA * 255);
                    }
                }
            }
        }
    }
    // Use drawImage so subsequent layers (SAM masks, prompts) can composite on top
    const tmp = document.createElement('canvas');
    tmp.width  = W;
    tmp.height = H;
    tmp.getContext('2d').putImageData(imgData, 0, 0);
    overlayCtx.drawImage(tmp, 0, 0);
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
    // Use drawImage so annotation masks drawn before are not wiped
    const tmp = document.createElement('canvas');
    tmp.width  = W;
    tmp.height = H;
    tmp.getContext('2d').putImageData(imgData, 0, 0);
    overlayCtx.drawImage(tmp, 0, 0);
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
    _updateBrushUI();  // show/hide "Init from SAM" button based on mask availability
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
    if (_me.promptType === 'brush') {
        setStatus('Segment is not available in brush mode');
        return;
    }
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

    if (_me.promptType === 'brush') {
        if (!_brushCanvas) { setStatus('No brush canvas'); return; }
        const brushMask = _extractBrushMask();
        const hasPaint = brushMask.some(row => row.some(v => v));
        if (!hasPaint) {
            setStatus('Brush canvas is empty — paint something first');
            return;
        }
        try {
            const resp = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_path:     _me.currentImagePath,
                    class_name:     _me.selectedClass,
                    instance_id:    _me.instanceId,
                    mask_data:      brushMask,
                    prompt_type:    'brush',
                    prompt_content: {},
                }),
            });
            if (!resp.ok) { setStatus('Save failed'); return; }
            _me.instanceId++;
            $id('instance-id-input').value = _me.instanceId;
            await loadAnnotations();
            setStatus('Brush annotation saved');
        } catch (err) {
            setStatus('Network error on save');
            console.error('[mask_editor] brush save error:', err);
        }
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
    _me.annotationMasks = {};
    await loadAllAnnotationMasks();
}

async function loadAllAnnotationMasks() {
    if (!_me.currentImagePath || _me.annotations.length === 0) return;
    const imagePath = _me.currentImagePath;
    await Promise.all(_me.annotations.map(async ann => {
        const key = `${ann.class}:${ann.instance_id}`;
        try {
            const params = new URLSearchParams({ class_name: ann.class, instance_id: ann.instance_id });
            const resp = await fetch(`/api/annotation/mask/${imagePath}?${params}`);
            if (resp.ok) {
                const data = await resp.json();
                _me.annotationMasks[key] = decodeMaskRLE(data.mask.shape, data.mask.rle);
                redrawOverlay();
            }
        } catch { /* mask load failure is non-fatal */ }
    }));
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

function selectAnnotation(ann) {
    // Toggle: clicking the already-highlighted annotation deselects it
    if (_me.highlightedAnnotation &&
        _me.highlightedAnnotation.class === ann.class &&
        _me.highlightedAnnotation.instance_id === ann.instance_id) {
        _me.highlightedAnnotation = null;
        renderAnnotationList();
        redrawOverlay();
        setStatus('Annotation deselected');
        return;
    }

    if (_me.classes.includes(ann.class)) {
        $id('class-select').value = ann.class;
        _me.selectedClass = ann.class;
    }
    _me.instanceId = ann.instance_id;
    $id('instance-id-input').value = ann.instance_id;

    // Clear any active SAM results so the canvas focuses on saved masks
    _me.currentMasks    = [];
    _me.selectedMaskIdx = 0;
    $id('mask-prev-container').innerHTML = '';

    _me.highlightedAnnotation = { class: ann.class, instance_id: ann.instance_id };
    renderAnnotationList();
    redrawOverlay();
    setStatus(`Highlighting: ${ann.class} #${ann.instance_id}`);
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

// ── Brush tool ─────────────────────────────────────────────────────────────

function _initBrushCanvas() {
    _brushCanvas = document.createElement('canvas');
    _brushCanvas.width  = imageCanvas.width;
    _brushCanvas.height = imageCanvas.height;
    _brushCtx = _brushCanvas.getContext('2d');
}

function _brushPaint(x, y, erase) {
    if (!_brushCtx) return;
    _brushCtx.save();
    _brushCtx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
    _brushCtx.fillStyle = 'rgba(255,255,255,1)';
    _brushCtx.beginPath();
    _brushCtx.arc(x, y, _me.brushSize, 0, Math.PI * 2);
    _brushCtx.fill();
    _brushCtx.restore();
}

function renderBrushMask() {
    if (!_brushCanvas) return;
    const [r, g, b] = classColor(_me.selectedClass);
    const scratch = document.createElement('canvas');
    scratch.width  = overlayCanvas.width;
    scratch.height = overlayCanvas.height;
    const sCtx = scratch.getContext('2d');
    sCtx.fillStyle = `rgba(${r},${g},${b},${160 / 255})`;
    sCtx.fillRect(0, 0, scratch.width, scratch.height);
    sCtx.globalCompositeOperation = 'destination-in';
    sCtx.drawImage(_brushCanvas, 0, 0);
    overlayCtx.drawImage(scratch, 0, 0);
}

function _drawBrushCursor(x, y) {
    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(0,0,0,0.5)';
    overlayCtx.lineWidth   = 1.5;
    overlayCtx.setLineDash([]);
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, _me.brushSize, 0, Math.PI * 2);
    overlayCtx.stroke();
    overlayCtx.strokeStyle = 'rgba(255,255,255,0.85)';
    overlayCtx.lineWidth   = 1;
    overlayCtx.setLineDash([4, 3]);
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, _me.brushSize, 0, Math.PI * 2);
    overlayCtx.stroke();
    overlayCtx.restore();
}

function _extractBrushMask() {
    const W = _brushCanvas.width;
    const H = _brushCanvas.height;
    const imgData = _brushCtx.getImageData(0, 0, W, H);
    const buf  = imgData.data;
    const mask = new Array(H);
    for (let row = 0; row < H; row++) {
        mask[row] = new Array(W);
        const rowBase = row * W * 4;
        for (let col = 0; col < W; col++) {
            mask[row][col] = buf[rowBase + col * 4 + 3] > 0;
        }
    }
    return mask;
}

function clearBrushCanvas() {
    if (!_brushCtx) return;
    _brushCtx.clearRect(0, 0, _brushCanvas.width, _brushCanvas.height);
    redrawOverlay();
    setStatus('Brush canvas cleared');
}

function initBrushFromSAMMask() {
    if (!_brushCtx || _me.currentMasks.length === 0) return;
    const mask = _me.currentMasks[_me.selectedMaskIdx];
    const H = mask.length;
    const W = H > 0 ? mask[0].length : 0;
    if (!H || !W) return;
    const imgData = _brushCtx.createImageData(W, H);
    const buf = imgData.data;
    for (let row = 0; row < H; row++) {
        const maskRow = mask[row];
        for (let col = 0; col < W; col++) {
            if (maskRow[col]) {
                const i = (row * W + col) * 4;
                buf[i] = buf[i + 1] = buf[i + 2] = 255;
                buf[i + 3] = 255;
            }
        }
    }
    _brushCtx.clearRect(0, 0, W, H);
    _brushCtx.putImageData(imgData, 0, 0);
    redrawOverlay();
    setStatus('Brush initialized from SAM mask');
}

function _updateBrushCursorStyle() {
    overlayCanvas.style.cursor = (_me.promptType === 'brush') ? 'none' : 'crosshair';
}

function _updateBrushUI() {
    const isBrush = _me.promptType === 'brush';
    const brushSection = $id('brush-section');
    if (brushSection) brushSection.style.display = isBrush ? 'block' : 'none';
    const initBtn = $id('init-from-sam-btn');
    if (initBtn) {
        initBtn.style.display = (isBrush && _me.currentMasks.length > 0) ? 'block' : 'none';
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
