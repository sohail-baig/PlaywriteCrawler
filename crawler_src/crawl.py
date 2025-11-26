import argparse
import csv
import json
import re
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Optional
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import urllib.parse

from dictWords import (
    accept_words,
    non_acceptable,
    reject_words,
    setting_words,
    more_setting_words,
    reject_setting_words,
    login_words,
    words,
)

def compile_word_patterns(keys: Iterable[str], langs: Optional[Iterable[str]] = None) -> List[re.Pattern]:
    """Compile regex patterns from word lists across multiple languages"""
    if langs is None:
        langs = words.keys()
    values: Set[str] = set()
    for lang in langs:
        mapping = words.get(lang, {})
        for k in keys:
            v = mapping.get(k)
            if v:
                values.add(v.lower())
    for k in keys:
        values.add(k.lower())
    if not values:
        return []
    return [re.compile(r"\b(" + "|".join(re.escape(v) for v in values) + r")\b", re.IGNORECASE)]

# Compile patterns for cookie consent detection
ACCEPT_RX = compile_word_patterns(accept_words)
ACCEPT_RX.append(re.compile(r"accetta[\s\u00A0]+e[\s\u00A0]+continua", re.IGNORECASE))
ACCEPT_RX.append(re.compile(r"accetta[\s\u00A0]+e[\s\u00A0]+chiudi", re.IGNORECASE))
ACCEPT_RX.append(re.compile(r"accetta[\s\u00A0]+tutt[oaie]", re.IGNORECASE))

REJECT_RX = compile_word_patterns(reject_words)
SETTINGS_RX = compile_word_patterns(setting_words + more_setting_words)
REJECT_SETTINGS_RX = compile_word_patterns(reject_setting_words)
NON_ACCEPTABLE_RX = re.compile("|".join(re.escape(w) for w in non_acceptable), re.IGNORECASE) if non_acceptable else re.compile(r"a^")

MAX_ELEMENTS_PER_SCOPE = 300

def text_matches_any(txt: str, patterns: List[re.Pattern]) -> bool:
    """Check if text matches any of the provided regex patterns"""
    return any(p.search(txt) for p in patterns)

def find_and_click_by_text(scope, include: List[re.Pattern], exclude: Optional[re.Pattern] = None) -> bool:
    """Find and click buttons/links matching inclusion patterns and not matching exclusion patterns"""
    try:
        # Expanded selectors to catch more button types
        elems = scope.locator("button, [role='button'], [role='link'], a, input[type='button'], input[type='submit'], div[class*='button'], span[class*='button']")
        n_total = elems.count()
        n = min(n_total, MAX_ELEMENTS_PER_SCOPE)
        
        for i in range(n):
            el = elems.nth(i)
            try:
                if not el.is_visible(timeout=100):
                    continue
            except:
                continue
            
            label = ""
            try:
                label = (el.inner_text() or "").strip()
            except:
                pass
            
            if not label:
                try:
                    label = (el.get_attribute("aria-label") or "").strip()
                except:
                    pass
            
            # Also check title attribute
            if not label:
                try:
                    label = (el.get_attribute("title") or "").strip()
                except:
                    pass
            
            # Check value attribute for input elements
            if not label:
                try:
                    label = (el.get_attribute("value") or "").strip()
                except:
                    pass
            
            if not label:
                continue
            
            if exclude and exclude.search(label):
                continue
            
            if text_matches_any(label, include):
                try:
                    el.click(timeout=2000, force=True)
                    return True
                except Exception as e:
                    # Try scrolling into view and clicking again
                    try:
                        el.scroll_into_view_if_needed(timeout=1000)
                        el.click(timeout=2000)
                        return True
                    except:
                        continue
    except Exception as e:
        pass
    return False

def try_all_frames(page, include: List[re.Pattern], exclude: Optional[re.Pattern] = None) -> bool:
    """Try to find and click in main page and all iframes"""
    if find_and_click_by_text(page, include, exclude):
        return True
    
    try:
        for fr in page.frames:
            if fr == page.main_frame:
                continue
            try:
                if find_and_click_by_text(fr, include, exclude):
                    return True
            except:
                continue
    except:
        pass
    
    return False

def normalize_url(url: str) -> str:
    """Add https:// scheme if missing"""
    parsed = urlparse(url)
    if not parsed.scheme:
        return "https://" + url
    return url

def smart_goto(page, url: str, timeout_ms: int = 90000) -> str:
    """Try multiple URL variants (https, with www, http) to successfully navigate"""
    base = url if urlparse(url).scheme else "https://" + url
    p = urlparse(base)
    host = p.hostname or ""
    
    candidates = []
    # Try https first
    candidates.append(p._replace(scheme="https").geturl())
    # Try with www prefix
    if host and not host.startswith("www."):
        candidates.append(p._replace(scheme="https", netloc="www."+host).geturl())
    # Try http
    candidates.append(p._replace(scheme="http").geturl())
    # Try http with www
    if host and not host.startswith("www."):
        candidates.append(p._replace(scheme="http", netloc="www."+host).geturl())
    
    last_error = None
    for c in candidates:
        try:
            page.goto(c, wait_until="domcontentloaded", timeout=timeout_ms)
            return c
        except Exception as e:
            last_error = e
            continue
    
    # If all attempts failed, raise the last error
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to navigate to {url}")

def safe_wait(page, ms: int, deadline: float) -> bool:
    """Wait for specified milliseconds, respecting deadline"""
    left = ms / 1000
    chunk = 0.25
    while left > 0:
        if time.time() >= deadline:
            return False
        w = min(chunk, left)
        try:
            page.wait_for_timeout(int(w * 1000))
        except:
            time.sleep(w)
        left -= w
    return True

def handle_accept(page, deadline: float, budget_ms: int = 3000) -> bool:
    """Handle cookie acceptance with multiple attempts"""
    try:
        page.wait_for_load_state("domcontentloaded", timeout=2000)
    except:
        pass
    
    end = min(deadline, time.time() + budget_ms/1000)
    clicked = False
    attempts = 0
    max_attempts = 5
    
    while time.time() < end and attempts < max_attempts:
        if try_all_frames(page, ACCEPT_RX, NON_ACCEPTABLE_RX):
            clicked = True
            attempts += 1
            safe_wait(page, 500, end)
            # Continue to handle multi-step dialogs
            continue
        if not safe_wait(page, 200, end):
            break
        attempts += 1
    
    return clicked

def handle_reject(page, deadline: float, budget_ms: int = 4000) -> bool:
    """Handle cookie rejection with settings navigation if needed"""
    try:
        page.wait_for_load_state("domcontentloaded", timeout=2000)
    except:
        pass
    
    end = min(deadline, time.time() + budget_ms/1000)
    attempts = 0
    max_attempts = 5
    
    # First try direct reject buttons
    while time.time() < end and attempts < max_attempts:
        if try_all_frames(page, REJECT_RX):
            safe_wait(page, 500, end)
            # Try again in case of multi-step
            try_all_frames(page, REJECT_RX)
            return True
        if not safe_wait(page, 200, end):
            break
        attempts += 1
    
    if time.time() >= end:
        return False
    
    # If direct reject didn't work, try opening settings
    opened = try_all_frames(page, SETTINGS_RX)
    if not opened:
        return False
    
    if not safe_wait(page, 800, end):
        return False
    
    # Try reject in settings
    if try_all_frames(page, REJECT_SETTINGS_RX) or try_all_frames(page, REJECT_RX):
        safe_wait(page, 300, end)
        return True
    
    return False

def load_disconnect_domains(path: str, categories: Set[str]) -> Set[str]:
    """Load blocked domains from Disconnect's blocklist JSON"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    out = set()
    cats = data.get("categories", data)
    cats_norm = {c.lower() for c in categories}
    
    for cat, providers in cats.items():
        if cat.lower() not in cats_norm:
            continue
        
        if isinstance(providers, list):
            for prov in providers:
                for _, svc in prov.items():
                    if isinstance(svc, dict):
                        for _, domains in svc.items():
                            if isinstance(domains, list):
                                for d in domains:
                                    d = d.strip(".").lower()
                                    if "." in d:
                                        out.add(d)
                    elif isinstance(svc, list):
                        for d in svc:
                            d = d.strip(".").lower()
                            if "." in d:
                                out.add(d)
        elif isinstance(providers, dict):
            for _, meta in providers.items():
                for d in meta.get("domains", []):
                    d = d.strip(".").lower()
                    if "." in d:
                        out.add(d)
                for rule in meta.get("rules", []):
                    h = rule.replace("^","").replace("$","").lstrip(".").lower()
                    if "." in h:
                        out.add(h)
    
    return out

def should_block(url: str, blocked: Set[str]) -> bool:
    """Check if URL should be blocked based on domain matching"""
    host = urllib.parse.urlparse(url).hostname or ""
    host = host.lower()
    return any(host == s or host.endswith("." + s) for s in blocked)

def crawl_site(pw, url: str, mode: str, blocked: Set[str], *,
               site_timeout: int = 120, use_video: bool = True,
               use_har: bool = True, debug: bool = False) -> None:
    """Crawl a single website with specified mode (accept/reject/block)"""
    
    parsed = urlparse(normalize_url(url))
    host = parsed.hostname or "website"
    
    # Create output directories
    if mode == "accept":
        base = Path("crawl_data_accept")
    elif mode == "reject":
        base = Path("crawl_data_reject")
    else:
        base = Path("crawl_data_block")
    base.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    har_path = base / f"{host}.har"
    pre = base / f"{host}_pre_consent.png"
    post = base / f"{host}_post_consent.png"
    vid = base / f"{host}.webm"

    start = time.time()
    deadline = start + site_timeout

    # Configure browser context
    ctx_args = {
        "viewport": {"width": 1280, "height": 800},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    if use_har:
        ctx_args["record_har_path"] = str(har_path)
        ctx_args["record_har_mode"] = "minimal"
    
    if use_video:
        ctx_args["record_video_dir"] = str(base)

    browser = None
    context = None
    page = None

    try:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(**ctx_args)

        # Set up request blocking for block mode
        if mode == "block" and blocked:
            def handler(route, req):
                try:
                    if should_block(req.url, blocked):
                        return route.abort()
                except:
                    pass
                return route.continue_()
            context.route("**/*", handler)

        page = context.new_page()
        page.set_default_timeout(5000)
        page.set_default_navigation_timeout(90000)

        # Navigate to URL
        remaining = max(5000, int((deadline - time.time())*1000))
        try:
            actual_url = smart_goto(page, url, timeout_ms=remaining)
            if debug:
                print(f"Successfully navigated to: {actual_url}")
        except Exception as e:
            if debug:
                print(f"Failed to navigate to {url}: {e}")
            return

        # Wait 10 seconds after page load
        safe_wait(page, 10_000, deadline)

        # Take pre-consent screenshot
        try:
            page.screenshot(path=str(pre), full_page=True, timeout=10000)
        except Exception as e:
            if debug:
                print(f"Failed to take pre-consent screenshot: {e}")
            pass

        # Handle consent based on mode
        if mode == "block":
            handle_accept(page, deadline, 3000)
        elif mode == "accept":
            handle_accept(page, deadline, 3000)
        else:  # reject
            handle_reject(page, deadline, 4000)

        # Wait 5 seconds after consent handling
        safe_wait(page, 5_000, deadline)

        # Scroll down in multiple steps (bot detection bypass)
        try:
            for scroll_step in range(12):
                if time.time() >= deadline:
                    break
                
                try:
                    page.evaluate("() => { window.scrollBy(0, window.innerHeight * 0.9); }")
                except:
                    pass
                
                safe_wait(page, 500, deadline)
                
                # Check if reached bottom
                try:
                    scroll_info = page.evaluate("() => ({y:window.scrollY, inner:window.innerHeight, height:document.body.scrollHeight})")
                    if scroll_info and scroll_info["y"] + scroll_info["inner"] + 2 >= scroll_info["height"]:
                        break
                except:
                    pass
        except Exception as e:
            if debug:
                print(f"Error during scrolling: {e}")
            pass

        # Wait 5 seconds after scrolling
        safe_wait(page, 5_000, deadline)

        # Take post-consent screenshot
        try:
            page.screenshot(path=str(post), full_page=True, timeout=10000)
        except Exception as e:
            if debug:
                print(f"Failed to take post-consent screenshot: {e}")
            pass

        # Save document.cookie
        try:
            cookies = page.evaluate("() => document.cookie")
            with open(base / f"{host}_document_cookie.json", "w", encoding="utf-8") as f:
                json.dump({"document_cookie": cookies}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if debug:
                print(f"Failed to save document.cookie: {e}")
            pass

    except Exception as e:
        if debug:
            print(f"Error crawling {url}: {e}")
    
    finally:
        # Clean up resources
        try:
            if page:
                page.close()
        except:
            pass
        
        try:
            if context:
                context.close()
        except:
            pass
        
        try:
            if browser:
                browser.close()
        except:
            pass

        # Rename video file
        if use_video:
            try:
                candidates = sorted(base.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    src = candidates[0]
                    if src.exists() and src != vid:
                        if vid.exists():
                            try:
                                vid.unlink()
                            except:
                                pass
                        src.rename(vid)
            except Exception as e:
                if debug:
                    print(f"Failed to rename video: {e}")
                pass

def read_sites_csv(path: str) -> List[Tuple[str, str]]:
    """Read list of sites from CSV file"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            u = row[0].strip()
            if not u or u.startswith("#"):
                continue
            # Extract country code if present
            c = row[1].strip() if len(row) > 1 else ""
            out.append((u, c))
    return out

def main():
    p = argparse.ArgumentParser(description="Web privacy crawler for cookie consent analysis")
    p.add_argument("-m", "--mode", choices=["accept", "reject", "block"], required=True,
                   help="Crawl mode: accept all cookies, reject all cookies, or block trackers")
    p.add_argument("-l", "--list", required=True,
                   help="CSV file with list of websites to crawl")
    p.add_argument("--disconnect", 
                   help="Path to Disconnect blocklist JSON (required for block mode)")
    p.add_argument("--site-timeout", type=int, default=120,
                   help="Timeout per site in seconds (default: 120)")
    p.add_argument("--no-video", action="store_true",
                   help="Disable video recording")
    p.add_argument("--no-har", action="store_true",
                   help="Disable HAR recording")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug output")
    a = p.parse_args()

    # Load sites list
    sites = read_sites_csv(a.list)
    print(f"Loaded {len(sites)} sites from {a.list}")

    # Load blocked domains for block mode
    blocked = set()
    if a.mode == "block":
        if not a.disconnect:
            print("Error: --disconnect argument is required for block mode")
            return
        
        cats = {
            "Advertising",
            "Analytics",
            "Social",
            "FingerprintingInvasive",
            "FingerprintingGeneral"
        }
        blocked = load_disconnect_domains(a.disconnect, cats)
        print(f"Loaded {len(blocked)} blocked domains from Disconnect list")

    # Perform crawls
    with sync_playwright() as pw:
        for idx, (raw, country) in enumerate(sites, 1):
            u = normalize_url(raw)
            print(f"[{idx}/{len(sites)}] Crawling {u} (mode={a.mode}, country={country})")
            
            try:
                crawl_site(
                    pw,
                    u,
                    a.mode,
                    blocked,
                    site_timeout=a.site_timeout,
                    use_video=not a.no_video,
                    use_har=not a.no_har,
                    debug=a.debug
                )
                print(f"  ✓ Completed {u}")
            except Exception as e:
                print(f"  ✗ Error crawling {u}: {e}")
                if a.debug:
                    import traceback
                    traceback.print_exc()

    print("Crawling complete!")

if __name__ == "__main__":
    main()