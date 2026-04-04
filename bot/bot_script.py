"""
MockShop Bot with unified behavioural logging schema.
Generates agent session logs with the same event model used by the frontend human logger.
"""

import argparse
import json
import math
import random
import string
import time
from datetime import datetime
from pathlib import Path
from urllib import error, request

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8001/demo"
API_LOG_ENDPOINT = "http://localhost:8001/api/logs"
VIEWPORT = {"width": 1280, "height": 800}

PRODUCT_CHECKLIST = [
    "SoundPeak Pro X",
    "UltraPhone 15",
    "SlimBook Air",
    "FitBand Pro",
    "BassCore 300",
    "MiniPhone Go",
    "ProPad Ultra",
    "AirBuds 3",
    "SmartWatch X2",
    "USB-C Hub Pro",
    "NoiseCanceller Elite",
    "MagCharge Pad",
]


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def ensure_backend_available(url):
    try:
        request.urlopen(f"{url}/index.html", timeout=5)
    except error.URLError as exc:
        raise SystemExit(
            "Backend/frontend not reachable at "
            f"{url}/index.html. Start backend first on port 8001. Error: {exc}"
        )


class SessionLogger:
    def __init__(self, session_type="agent", iteration_id=None):
        self.session_type = "agent" if session_type == "agent" else "human"
        self.session_id = self._build_session_id(self.session_type, iteration_id=iteration_id)
        self.start_perf = time.perf_counter()
        self.events = []
        self.event_id = 0

    @staticmethod
    def _build_session_id(session_type, iteration_id=None):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        if iteration_id is None:
            return f"{session_type}_{now}_{uid}"
        return f"{session_type}_{now}_{uid}_iter{iteration_id}"

    def now_ms(self):
        return int((time.perf_counter() - self.start_perf) * 1000)

    def log_event(self, event_type, data):
        self.event_id += 1
        self.events.append(
            {
                "event_id": self.event_id,
                "timestamp_ms": self.now_ms(),
                "type": event_type,
                "data": data,
            }
        )

    def to_dict(self, user_agent, viewport):
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "start_time_ms": 0,
            "end_time_ms": self.now_ms(),
            "user_agent": user_agent,
            "viewport": viewport,
            "events": self.events,
        }

    def save(self, log_dir, user_agent, viewport, filename=None):
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(user_agent=user_agent, viewport=viewport)
        out_path = log_dir / (filename if filename else f"{self.session_id}.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Bot] Local log saved: {out_path}")
        return payload, out_path


class BotActor:
    def __init__(self, page, logger):
        self.page = page
        self.logger = logger
        self.cursor_x = VIEWPORT["width"] / 2
        self.cursor_y = VIEWPORT["height"] / 2

    def _selector_for_locator(self, locator, fallback=None):
        try:
            handle = locator.element_handle()
            if not handle:
                return fallback
            selector = handle.evaluate(
                """el => {
                    if (el.dataset && el.dataset.logId) return `[data-log-id='${el.dataset.logId}']`;
                    if (el.id) return `#${el.id}`;
                    return el.tagName ? el.tagName.toLowerCase() : null;
                }"""
            )
            return selector or fallback
        except Exception:
            return fallback

    @staticmethod
    def _bbox_for_log(bbox):
        return {
            "x": int(round(bbox["x"])),
            "y": int(round(bbox["y"])),
            "width": int(round(bbox["width"])),
            "height": int(round(bbox["height"])),
        }

    def log_navigation(self, url, referrer):
        self.logger.log_event(
            "navigation",
            {
                "url": url,
                "referrer": referrer or "",
            },
        )

    def goto_with_log(self, path, referrer=""):
        url = f"{BASE_URL}/{path}?session_type=agent"
        self.page.goto(url, wait_until="load")
        self.log_navigation(url=self.page.url, referrer=referrer)
        return url

    def smooth_move_to(self, end_x, end_y, target_selector=None, target_bbox=None):
        sx = self.cursor_x
        sy = self.cursor_y
        dx = end_x - sx
        dy = end_y - sy
        distance = math.sqrt(dx * dx + dy * dy)
        steps = int(clamp(distance / 18, 8, 40))
        duration_s = random.uniform(0.20, 0.70)

        path = []
        for i in range(steps + 1):
            t = i / steps
            ease = t * t * (3 - 2 * t)
            x = sx + dx * ease
            y = sy + dy * ease

            jitter_scale = max(0.2, 1.0 - abs(0.5 - t) * 2)
            x += random.gauss(0, 0.9) * jitter_scale
            y += random.gauss(0, 0.9) * jitter_scale

            self.page.mouse.move(x, y)
            ts = self.logger.now_ms()
            path.append({"x": int(round(x)), "y": int(round(y)), "t_ms": ts})
            if i < steps:
                time.sleep(duration_s / steps)

        self.cursor_x = end_x
        self.cursor_y = end_y

        if len(path) >= 2:
            self.logger.log_event(
                "mousemove",
                {
                    "path": path,
                    "target_element": target_selector,
                    "target_bbox": target_bbox,
                },
            )

    def gaussian_target_point(self, bbox):
        cx = bbox["x"] + bbox["width"] / 2
        cy = bbox["y"] + bbox["height"] / 2
        sigma_x = max(2.0, bbox["width"] * 0.12)
        sigma_y = max(2.0, bbox["height"] * 0.12)

        x = random.gauss(cx, sigma_x)
        y = random.gauss(cy, sigma_y)

        x = clamp(x, bbox["x"] + 1, bbox["x"] + bbox["width"] - 1)
        y = clamp(y, bbox["y"] + 1, bbox["y"] + bbox["height"] - 1)
        return x, y

    def click_locator(self, locator, fallback_selector=None):
        if locator.count() == 0:
            return False

        box = locator.bounding_box()
        if not box:
            return False

        selector = self._selector_for_locator(locator, fallback=fallback_selector)
        bbox_for_log = self._bbox_for_log(box)

        tx, ty = self.gaussian_target_point(box)
        self.smooth_move_to(tx, ty, target_selector=selector, target_bbox=bbox_for_log)

        button = "left"
        down_ts = self.logger.now_ms()
        self.logger.log_event(
            "mousedown",
            {
                "x": int(round(tx)),
                "y": int(round(ty)),
                "target_element": selector,
                "button": button,
            },
        )

        self.page.mouse.down()
        hold_ms = int(clamp(random.gauss(105, 30), 35, 260))
        time.sleep(hold_ms / 1000.0)
        self.page.mouse.up()

        up_ts = self.logger.now_ms()
        self.logger.log_event(
            "mouseup",
            {
                "x": int(round(tx)),
                "y": int(round(ty)),
                "target_element": selector,
                "button": button,
                "hold_duration_ms": max(0, up_ts - down_ts),
            },
        )

        cx = bbox_for_log["x"] + bbox_for_log["width"] / 2
        cy = bbox_for_log["y"] + bbox_for_log["height"] / 2
        self.logger.log_event(
            "click",
            {
                "x": int(round(tx)),
                "y": int(round(ty)),
                "target_element": selector,
                "target_bbox": bbox_for_log,
                "offset_from_center": {
                    "dx": int(round(tx - cx)),
                    "dy": int(round(ty - cy)),
                },
            },
        )
        return True

    def scroll_pattern(self, steps=None):
        steps = steps or random.randint(3, 7)
        start_y = int(round(self.page.evaluate("() => window.scrollY")))
        start_ts = self.logger.now_ms()
        entries = []

        for _ in range(steps):
            delta = int(clamp(random.gauss(180, 70), 35, 420))
            self.page.mouse.wheel(0, delta)
            time.sleep(random.uniform(0.08, 0.22))
            y = int(round(self.page.evaluate("() => window.scrollY")))
            entries.append({"y": y, "t_ms": self.logger.now_ms()})

        # Occasional correction movement for overshoot profile.
        if random.random() < 0.35:
            corr = -int(clamp(abs(random.gauss(90, 35)), 25, 220))
            self.page.mouse.wheel(0, corr)
            time.sleep(random.uniform(0.08, 0.20))
            y = int(round(self.page.evaluate("() => window.scrollY")))
            entries.append({"y": y, "t_ms": self.logger.now_ms()})

        end_y = int(round(self.page.evaluate("() => window.scrollY")))
        self.logger.log_event(
            "scroll",
            {
                "start_y": start_y,
                "end_y": end_y,
                "duration_ms": max(0, self.logger.now_ms() - start_ts),
                "steps": entries,
            },
        )

    def _predict_value_after_key(self, before, key):
        if key == "Backspace":
            return before[:-1]
        if len(key) == 1:
            return (before + key)[:50]
        return before[:50]

    def _press_key(self, locator, selector, key):
        try:
            before = (locator.input_value() or "")[:50]
        except Exception:
            before = ""

        after = self._predict_value_after_key(before, key)
        self.logger.log_event(
            "keydown",
            {
                "target_element": selector,
                "key": key,
                "value_before": before,
                "value_after": after,
            },
        )

        self.page.keyboard.down(key)
        hold_ms = int(clamp(random.gauss(92, 28), 30, 220))
        time.sleep(hold_ms / 1000.0)
        self.page.keyboard.up(key)

        self.logger.log_event(
            "keyup",
            {
                "target_element": selector,
                "key": key,
                "hold_duration_ms": hold_ms,
            },
        )

    def type_into(self, selector, value):
        locator = self.page.locator(selector)
        if locator.count() == 0:
            return False

        self.click_locator(locator, fallback_selector=selector)
        time.sleep(random.uniform(0.05, 0.20))

        resolved_selector = self._selector_for_locator(locator, fallback=selector)

        for ch in value:
            # Small typo-correction probability to emulate human-like error repair.
            if random.random() < 0.08 and ch.isalpha():
                wrong = random.choice("abcdefghijklmnopqrstuvwxyz")
                self._press_key(locator, resolved_selector, wrong)
                time.sleep(random.uniform(0.05, 0.13))
                self._press_key(locator, resolved_selector, "Backspace")
                time.sleep(random.uniform(0.05, 0.14))

            self._press_key(locator, resolved_selector, ch)
            time.sleep(random.uniform(0.06, 0.23))

        return True


def pick_targets(n=None):
    n = n or random.randint(1, 2)
    return random.sample(PRODUCT_CHECKLIST, min(n, len(PRODUCT_CHECKLIST)))


def navigate_listings(actor, target_product):
    print(f"[Bot] Navigating listings for: {target_product}")
    index_url = actor.goto_with_log("index.html", referrer="")
    time.sleep(random.uniform(0.5, 1.2))
    actor.scroll_pattern(steps=random.randint(2, 5))

    cards = actor.page.locator(".product-card")
    total = cards.count()
    if total == 0:
        print("[Bot] No product cards found.")
        return False

    chosen = None
    for i in range(total):
        card = cards.nth(i)
        text = (card.inner_text() or "").lower()
        if target_product.lower() in text:
            chosen = card
            break

    if chosen is None:
        chosen = cards.nth(random.randint(0, total - 1))

    if not actor.click_locator(chosen, fallback_selector=".product-card"):
        return False

    actor.page.wait_for_load_state("domcontentloaded")
    actor.log_navigation(actor.page.url, referrer=index_url)
    return True


def navigate_product(actor):
    print("[Bot] Product page interactions")
    time.sleep(random.uniform(0.3, 0.9))
    actor.scroll_pattern(steps=random.randint(2, 4))

    if random.random() < 0.35:
        plus_btn = actor.page.locator("[data-log-id='qty-plus']")
        clicks = random.randint(1, 2)
        for _ in range(clicks):
            actor.click_locator(plus_btn, fallback_selector="[data-log-id='qty-plus']")
            time.sleep(random.uniform(0.12, 0.35))

    add_btn = actor.page.locator("[data-log-id='add-to-cart']")
    actor.click_locator(add_btn, fallback_selector="[data-log-id='add-to-cart']")
    time.sleep(random.uniform(0.4, 1.0))


def navigate_cart(actor):
    print("[Bot] Cart page interactions")
    cart_ref = actor.page.url
    actor.goto_with_log("cart.html", referrer=cart_ref)
    time.sleep(random.uniform(0.4, 1.0))
    actor.scroll_pattern(steps=random.randint(1, 3))

    if random.random() < 0.45:
        promo = random.choice(["SAVE10", "HELLO5", "DEAL20"])
        actor.type_into("[data-log-id='promo-input']", promo)
        time.sleep(random.uniform(0.1, 0.3))
        actor.click_locator(actor.page.locator("[data-log-id='promo-apply']"), "[data-log-id='promo-apply']")
        time.sleep(random.uniform(0.2, 0.5))

    actor.click_locator(actor.page.locator("[data-log-id='proceed-checkout']"), "[data-log-id='proceed-checkout']")
    actor.page.wait_for_load_state("domcontentloaded")
    actor.log_navigation(actor.page.url, referrer=f"{BASE_URL}/cart.html?session_type=agent")


def navigate_checkout(actor):
    print("[Bot] Checkout form fill")
    first_name = random.choice(["Alex", "Jordan", "Morgan", "Taylor"])
    last_name = random.choice(["Singh", "Smith", "Patel", "Brown"])
    shipping_data = {
        "[data-log-id='input-firstname']": first_name,
        "[data-log-id='input-lastname']": last_name,
        "[data-log-id='input-email']": f"user{random.randint(100, 999)}@example.test",
        "[data-log-id='input-address']": f"{random.randint(10, 999)} Test Street",
        "[data-log-id='input-city']": random.choice(["Austin", "Seattle", "Chicago", "Boston"]),
        "[data-log-id='input-zip']": str(random.randint(10000, 99999)),
        "[data-log-id='input-state']": random.choice(["TX", "CA", "NY", "FL"]),
    }

    for selector, value in shipping_data.items():
        actor.type_into(selector, value)
        time.sleep(random.uniform(0.08, 0.25))

    actor.click_locator(actor.page.locator("[data-log-id='continue-to-payment']"), "[data-log-id='continue-to-payment']")
    time.sleep(random.uniform(0.4, 0.8))

    card_data = {
        "[data-log-id='input-cardname']": f"{first_name} {last_name}",
        "[data-log-id='input-cardnum']": "4111111111111111",
        "[data-log-id='input-expm']": str(random.randint(1, 12)).zfill(2),
        "[data-log-id='input-expy']": str(random.randint(26, 30)),
        "[data-log-id='input-cvv']": str(random.randint(100, 999)),
    }

    for selector, value in card_data.items():
        actor.type_into(selector, value)
        time.sleep(random.uniform(0.08, 0.22))

    actor.click_locator(actor.page.locator("[data-log-id='review-order']"), "[data-log-id='review-order']")
    time.sleep(random.uniform(0.5, 1.1))
    actor.click_locator(actor.page.locator("[data-log-id='place-order']"), "[data-log-id='place-order']")
    time.sleep(random.uniform(0.8, 1.6))


def post_session_to_backend(payload):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        API_LOG_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=8) as resp:
            print(f"[Bot] Backend save response: {resp.status}")
    except Exception as exc:
        print(f"[Bot] Backend save skipped (local file still saved): {exc}")


def run_bot_session(targets=None, iterations=3, push_backend=False):
    print(f"\n[Bot] Session start. Purchase iterations: {iterations}")

    ensure_backend_available(BASE_URL)
    bot_log_dir = Path(__file__).resolve().parent.parent / "logs" / "bot_logs"
    bot_log_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())

        for iteration_index in range(iterations):
            iteration_id = iteration_index + 1
            iteration_targets = targets or pick_targets()
            logger = SessionLogger(session_type="agent", iteration_id=iteration_id)
            actor = BotActor(page=page, logger=logger)
            print(
                f"\n[Bot] Purchase iteration {iteration_id}/{iterations} "
                f"targets: {iteration_targets}"
            )

            for product in iteration_targets:
                if navigate_listings(actor, product):
                    navigate_product(actor)
                    time.sleep(random.uniform(0.2, 0.6))

            navigate_cart(actor)
            navigate_checkout(actor)

            user_agent = page.evaluate("() => navigator.userAgent")
            payload, _ = logger.save(
                log_dir=bot_log_dir,
                user_agent=user_agent,
                viewport=VIEWPORT,
                filename=f"{iteration_id}.json",
            )
            if push_backend:
                post_session_to_backend(payload)

            if iteration_index < iterations - 1:
                time.sleep(random.uniform(0.8, 2.0))

        browser.close()

    print("[Bot] Session complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Number of bot sessions")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of complete purchase iterations inside each session",
    )
    parser.add_argument("--products", nargs="*", help="Optional product names to target")
    parser.add_argument(
        "--push-backend",
        action="store_true",
        help="Also POST each iteration log to backend /api/logs (creates additional files in logs root)",
    )
    args = parser.parse_args()

    for i in range(args.runs):
        print(f"\n[Bot] Run {i + 1}/{args.runs}")
        run_bot_session(
            targets=args.products,
            iterations=max(1, args.iterations),
            push_backend=args.push_backend,
        )
        if i < args.runs - 1:
            time.sleep(random.uniform(1.5, 4.0))


if __name__ == "__main__":
    main()
