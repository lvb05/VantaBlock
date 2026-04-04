// Unified behavioural logger for human and agent sessions.
(function () {
    const PAGE = window.PAGE_NAME || "unknown";
    const API_ENDPOINT = "http://localhost:8001/api/logs";
    const STORAGE_KEY = "unified_behaviour_session";
    const FLUSH_INTERVAL_MS = 30000;
    const SESSION_STALE_MS = 30 * 60 * 1000;

    const pageMonotonicEpoch = Date.now() - performance.now();
    const nowMonotonicMs = () => Math.round(pageMonotonicEpoch + performance.now());

    const urlType = new URLSearchParams(window.location.search).get("session_type");
    const requestedType = (urlType || window.SESSION_TYPE || "human").toLowerCase() === "agent" ? "agent" : "human";

    function pad2(n) {
        return String(n).padStart(2, "0");
    }

    function makeSessionId(sessionType) {
        const d = new Date();
        const stamp = `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}_${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
        const uid = Math.random().toString(36).slice(2, 8);
        const prefix = sessionType === "agent" ? "agent" : "human";
        return `${prefix}_${stamp}_${uid}`;
    }

    function loadState() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return null;
            const parsed = JSON.parse(raw);
            if (!parsed || !Array.isArray(parsed.events)) return null;
            return parsed;
        } catch (_) {
            return null;
        }
    }

    function saveState() {
        state.last_updated_wall_ms = Date.now();
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }

    function newState(sessionType) {
        return {
            session_id: makeSessionId(sessionType),
            session_type: sessionType,
            start_mono_ms: null,
            start_time_ms: 0,
            end_time_ms: 0,
            next_event_id: 1,
            events: [],
            user_agent: navigator.userAgent,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight,
            },
            last_updated_wall_ms: Date.now(),
        };
    }

    let state = loadState();
    const isStale = state && Date.now() - (state.last_updated_wall_ms || 0) > SESSION_STALE_MS;
    if (!state || isStale) {
        state = newState(requestedType);
    } else if (urlType) {
        // Explicit URL session_type takes precedence when provided.
        state.session_type = requestedType;
        if (!state.session_id.startsWith(`${requestedType}_`)) {
            state.session_id = makeSessionId(requestedType);
            state.start_mono_ms = null;
            state.next_event_id = 1;
            state.events = [];
            state.end_time_ms = 0;
        }
    }

    function elementSelector(el) {
        if (!(el instanceof Element)) return null;
        if (el.dataset && el.dataset.logId) return `[data-log-id='${el.dataset.logId}']`;
        if (el.id) return `#${el.id}`;
        return el.tagName.toLowerCase();
    }

    function elementBBox(el) {
        if (!(el instanceof Element)) return null;
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        return {
            x: Math.round(rect.left),
            y: Math.round(rect.top),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
        };
    }

    function toSessionTs(absMonoMs) {
        if (state.start_mono_ms == null) {
            state.start_mono_ms = absMonoMs;
        }
        const ts = Math.max(0, Math.round(absMonoMs - state.start_mono_ms));
        state.end_time_ms = ts;
        return ts;
    }

    function logEvent(type, data, absMonoMs) {
        const nowAbs = absMonoMs == null ? nowMonotonicMs() : absMonoMs;
        const ts = toSessionTs(nowAbs);
        state.events.push({
            event_id: state.next_event_id,
            timestamp_ms: ts,
            type,
            data,
        });
        state.next_event_id += 1;
        saveState();
    }

    // Log a navigation event on every page load.
    logEvent("navigation", {
        url: window.location.href,
        referrer: document.referrer || "",
        page: PAGE,
    });

    const mouseButtonName = {
        0: "left",
        1: "middle",
        2: "right",
    };

    let lastMousePoint = null;
    let lastMouseEmit = 0;
    const mouseDownByButton = {};

    document.addEventListener("mousemove", (e) => {
        const nowAbs = nowMonotonicMs();
        const current = { x: e.clientX, y: e.clientY, abs: nowAbs };

        if (!lastMousePoint) {
            lastMousePoint = current;
            return;
        }

        const dt = current.abs - lastMouseEmit;
        const dx = current.x - lastMousePoint.x;
        const dy = current.y - lastMousePoint.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dt < 50 && dist <= 10) {
            return;
        }

        const target = document.elementFromPoint(current.x, current.y);
        const selector = elementSelector(target);
        const bbox = selector ? elementBBox(target) : null;

        const prevTs = toSessionTs(lastMousePoint.abs);
        const currTs = toSessionTs(current.abs);

        logEvent("mousemove", {
            path: [
                { x: Math.round(lastMousePoint.x), y: Math.round(lastMousePoint.y), t_ms: prevTs },
                { x: Math.round(current.x), y: Math.round(current.y), t_ms: currTs },
            ],
            target_element: selector,
            target_bbox: bbox,
        }, current.abs);

        lastMousePoint = current;
        lastMouseEmit = current.abs;
    }, { passive: true });

    document.addEventListener("mousedown", (e) => {
        const nowAbs = nowMonotonicMs();
        const selector = elementSelector(e.target);
        mouseDownByButton[e.button] = {
            t_ms: toSessionTs(nowAbs),
            x: e.clientX,
            y: e.clientY,
            target_element: selector,
        };

        logEvent("mousedown", {
            x: Math.round(e.clientX),
            y: Math.round(e.clientY),
            target_element: selector,
            button: mouseButtonName[e.button] || "unknown",
        }, nowAbs);
    }, true);

    document.addEventListener("mouseup", (e) => {
        const nowAbs = nowMonotonicMs();
        const selector = elementSelector(e.target);
        const down = mouseDownByButton[e.button];
        const nowTs = toSessionTs(nowAbs);
        const hold = down ? Math.max(0, nowTs - down.t_ms) : 0;

        logEvent("mouseup", {
            x: Math.round(e.clientX),
            y: Math.round(e.clientY),
            target_element: selector,
            button: mouseButtonName[e.button] || "unknown",
            hold_duration_ms: hold,
        }, nowAbs);

        delete mouseDownByButton[e.button];
    }, true);

    document.addEventListener("click", (e) => {
        const nowAbs = nowMonotonicMs();
        const selector = elementSelector(e.target);
        const bbox = elementBBox(e.target);

        let dx = 0;
        let dy = 0;
        if (bbox) {
            const cx = bbox.x + bbox.width / 2;
            const cy = bbox.y + bbox.height / 2;
            dx = Math.round(e.clientX - cx);
            dy = Math.round(e.clientY - cy);
        }

        logEvent("click", {
            x: Math.round(e.clientX),
            y: Math.round(e.clientY),
            target_element: selector,
            target_bbox: bbox,
            offset_from_center: { dx, dy },
        }, nowAbs);
    }, true);

    let lastKnownScrollY = Math.round(window.scrollY);
    let lastScrollSample = { t_abs: 0, y: lastKnownScrollY };
    let activeScroll = null;
    let finalizeScrollTimer = null;

    function finalizeScrollEvent(forceAbs) {
        if (!activeScroll) return;

        const endY = Math.round(window.scrollY);
        const endAbs = forceAbs == null ? nowMonotonicMs() : forceAbs;
        const duration = Math.max(0, toSessionTs(endAbs) - activeScroll.start_t_ms);
        const steps = activeScroll.steps.length > 0
            ? activeScroll.steps
            : [{ y: endY, t_ms: toSessionTs(endAbs) }];

        logEvent("scroll", {
            start_y: activeScroll.start_y,
            end_y: endY,
            duration_ms: duration,
            steps,
        }, endAbs);

        activeScroll = null;
    }

    window.addEventListener("scroll", () => {
        const nowAbs = nowMonotonicMs();
        const y = Math.round(window.scrollY);

        if (!activeScroll) {
            const startTs = toSessionTs(nowAbs);
            activeScroll = {
                start_y: lastKnownScrollY,
                start_t_ms: startTs,
                steps: [{ y, t_ms: startTs }],
            };
            lastScrollSample = { t_abs: nowAbs, y };
        } else {
            const dt = nowAbs - lastScrollSample.t_abs;
            const dy = Math.abs(y - lastScrollSample.y);
            if (dt >= 50 || dy >= 30) {
                activeScroll.steps.push({ y, t_ms: toSessionTs(nowAbs) });
                lastScrollSample = { t_abs: nowAbs, y };
            }
        }

        lastKnownScrollY = y;

        if (finalizeScrollTimer) {
            clearTimeout(finalizeScrollTimer);
        }
        finalizeScrollTimer = setTimeout(() => finalizeScrollEvent(), 150);
    }, { passive: true });

    const keyDownMap = {};

    function valueForInput(el) {
        if (!(el instanceof Element)) return "";
        if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
            return (el.value || "").slice(0, 50);
        }
        if (el instanceof HTMLSelectElement) {
            return (el.value || "").slice(0, 50);
        }
        return "";
    }

    function predictedValueAfterKey(before, key) {
        if (key === "Backspace") return before.slice(0, -1);
        if (key === "Delete" || key === "Enter" || key === "Tab") return before;
        if (key.length === 1) return (before + key).slice(0, 50);
        return before;
    }

    document.addEventListener("keydown", (e) => {
        const nowAbs = nowMonotonicMs();
        const selector = elementSelector(e.target);
        const before = valueForInput(e.target);
        const after = predictedValueAfterKey(before, e.key);

        logEvent("keydown", {
            target_element: selector,
            key: e.key,
            value_before: before,
            value_after: after,
        }, nowAbs);

        keyDownMap[`${selector || "unknown"}:${e.code}:${e.key}`] = toSessionTs(nowAbs);
    }, true);

    document.addEventListener("keyup", (e) => {
        const nowAbs = nowMonotonicMs();
        const selector = elementSelector(e.target);
        const keyId = `${selector || "unknown"}:${e.code}:${e.key}`;
        const downTs = keyDownMap[keyId];
        const upTs = toSessionTs(nowAbs);
        const hold = downTs == null ? 0 : Math.max(0, upTs - downTs);

        logEvent("keyup", {
            target_element: selector,
            key: e.key,
            hold_duration_ms: hold,
        }, nowAbs);

        delete keyDownMap[keyId];
    }, true);

    function buildSessionPayload() {
        return {
            session_id: state.session_id,
            session_type: state.session_type,
            start_time_ms: 0,
            end_time_ms: state.end_time_ms,
            user_agent: state.user_agent,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight,
            },
            events: state.events,
        };
    }

    function sendSession(keepalive) {
        const payload = buildSessionPayload();
        return fetch(API_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            keepalive: Boolean(keepalive),
        }).catch((err) => {
            console.warn("[logger] failed to send session", err);
        });
    }

    setInterval(() => {
        if (state.events.length > 0) {
            sendSession(false);
        }
    }, FLUSH_INTERVAL_MS);

    function flushAndSaveOnExit() {
        if (finalizeScrollTimer) {
            clearTimeout(finalizeScrollTimer);
        }
        finalizeScrollEvent(nowMonotonicMs());
        saveState();
        sendSession(true);
    }

    window.addEventListener("beforeunload", flushAndSaveOnExit);
    window.flushLogs = flushAndSaveOnExit;
})();