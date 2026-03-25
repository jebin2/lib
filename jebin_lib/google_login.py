import os
import json
import time
import getpass
import tempfile
import subprocess
import urllib.request
from custom_logger import logger_config


def _upload_screenshot_to_hf(page, hf_screenshot_repo_id, label="screenshot"):
    """Upload a page screenshot to HuggingFace for remote debugging."""
    if not hf_screenshot_repo_id:
        logger_config.warning("[GoogleLogin] hf_screenshot_repo_id not set — skipping screenshot upload.")
        return
    try:
        from jebin_lib.hf_dataset_client import HFDatasetClient
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        page.screenshot(path=tmp_path)
        HFDatasetClient(repo_id=hf_screenshot_repo_id).upload(tmp_path, f"screenshots/{label}.png")
        os.unlink(tmp_path)
        logger_config.info(f"[GoogleLogin] Screenshot uploaded: screenshots/{label}.png")
    except Exception as e:
        logger_config.error(f"[GoogleLogin] Failed to upload screenshot to HF: {e}")


class GoogleLoginAutomator:
    """Unified Google login and OAuth flow automator.

    Handles:
    - Account chooser (data-identifier / data-email / 'Use another account')
    - Email and password entry
    - 2-Step Verification wait (with mandatory HF screenshot uploads)
    - OAuth brand-account selection and consent screens
    - Auth code URL capture via CDP first, xdotool+xclip Docker fallback

    Credential priority: constructor args > env vars (GOOGLE_EMAIL / GOOGLE_PASSWORD) > interactive prompt

    Usage — login on an existing page (e.g. chat_bot_ui_handler):
        automator = GoogleLoginAutomator(hf_screenshot_repo_id="user/repo")
        automator.login(page)

    Usage — full OAuth round-trip (e.g. youtube_auto_pub):
        automator = GoogleLoginAutomator(
            authorization_code_path="/tmp/auth_code.txt",
            browser_profile_path="/path/to/profile",
            docker_name="my_container",
            hf_screenshot_repo_id="user/repo",
        )
        automator.authorize_oauth(auth_url, browser_config)
    """

    def __init__(
        self,
        email=None,
        password=None,
        browser_profile_path=None,
        authorization_code_path=None,
        docker_name=None,
        hf_screenshot_repo_id=None,
    ):
        self.email = email or os.getenv("GOOGLE_EMAIL") or os.getenv("OAUTH_EMAIL")
        self.password = password or os.getenv("GOOGLE_PASSWORD") or os.getenv("OAUTH_PASSWORD")
        self.browser_profile_path = browser_profile_path
        self.authorization_code_path = authorization_code_path
        self.docker_name = docker_name
        self.hf_screenshot_repo_id = hf_screenshot_repo_id
        self._cdp_port = 9224  # updated by authorize_oauth when browser_config is provided

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def login(self, page):
        """Automate Google login on an already-open browser page.

        Covers: account chooser → email → password → 2FA.
        Safe to call when the page is already authenticated — each step
        checks visibility before acting.

        Args:
            page: Playwright page object
        """
        def _safe_dialog_handler(dialog):
            try:
                dialog.accept()
            except Exception:
                pass

        page.on("dialog", _safe_dialog_handler)

        logger_config.info(f"[GoogleLogin] Step 1: Starting login. URL: {page.url}")
        page.wait_for_timeout(2000)

        self._handle_account_chooser(page)
        self._enter_email(page)
        self._enter_password(page)

        logger_config.info("[GoogleLogin] Step 5: Waiting after password submission...")
        page.wait_for_timeout(5000)
        self._upload_screenshot(page, "post_login")
        self._handle_2fa(page)

    def authorize_oauth(self, auth_url, browser_config):
        """Full OAuth round-trip: open browser → login → consent → capture auth code URL.

        Args:
            auth_url: The OAuth authorization URL to open
            browser_config: BrowserConfig instance (caller configures browser settings)

        Returns:
            True if auth code URL was captured and saved successfully
        """
        from browser_manager import BrowserManager

        self._clear_cookies()
        browser_config.url = auth_url
        self._cdp_port = getattr(browser_config, "debug_port", 9224)

        with BrowserManager(browser_config) as page:
            page.wait_for_timeout(8000)

            # If Google shows account chooser directly (cached session), skip login
            heading = page.query_selector("#headingText")
            if not (heading and heading.text_content() == "Choose your account or a brand account"):
                self.login(page)

            return self._handle_oauth_consent(page)

    # ------------------------------------------------------------------
    # Login steps
    # ------------------------------------------------------------------

    def _handle_account_chooser(self, page):
        logger_config.info("[GoogleLogin] Step 2: Checking for account chooser...")
        try:
            saved = page.locator(f'[data-identifier="{self.email}"]')
            saved_alt = page.locator(f'[data-email="{self.email}"]')
            use_another = page.get_by_text("Use another account", exact=True)

            if saved.is_visible():
                logger_config.info("[GoogleLogin] Step 2a: Clicking saved account (data-identifier).")
                saved.first.click()
                page.wait_for_timeout(2000)
            elif saved_alt.is_visible():
                logger_config.info("[GoogleLogin] Step 2b: Clicking saved account (data-email).")
                saved_alt.first.click()
                page.wait_for_timeout(2000)
            elif use_another.is_visible():
                logger_config.info("[GoogleLogin] Step 2c: Clicking 'Use another account'.")
                use_another.first.click()
                page.wait_for_timeout(2000)
            else:
                logger_config.info("[GoogleLogin] Step 2d: No account chooser visible.")
        except Exception as e:
            logger_config.error(f"[GoogleLogin] Account chooser error: {e}")

    def _enter_email(self, page):
        logger_config.info("[GoogleLogin] Step 3: Checking for email input...")
        try:
            page.wait_for_selector("#identifierId", timeout=5000)
        except Exception:
            pass
        if page.locator("#identifierId").is_visible():
            email, _ = self._get_credentials()
            logger_config.info("[GoogleLogin] Step 3a: Entering email.")
            page.wait_for_timeout(1000)
            page.fill("#identifierId", email)
            page.wait_for_selector("#identifierNext")
            page.wait_for_timeout(1000)
            page.click("#identifierNext")
        else:
            logger_config.info("[GoogleLogin] Step 3b: Email input not visible, skipping.")

    def _enter_password(self, page):
        logger_config.info("[GoogleLogin] Step 4: Checking for password input...")
        # Wait briefly in case the page is still transitioning (e.g. after account chooser click)
        try:
            page.wait_for_selector('input[type="password"]', timeout=5000)
        except Exception:
            pass
        if page.locator('input[type="password"]').is_visible():
            _, password = self._get_credentials()
            logger_config.info("[GoogleLogin] Step 4a: Entering password.")
            page.wait_for_timeout(1000)
            page.fill('input[type="password"]', password)
            page.wait_for_selector("#passwordNext")
            page.wait_for_timeout(1000)
            page.click("#passwordNext")
        else:
            logger_config.info("[GoogleLogin] Step 4b: Password input not visible, skipping.")

    def _handle_2fa(self, page):
        """Poll for 2FA screen, uploading a screenshot each iteration until resolved."""
        logger_config.info("[GoogleLogin] Step 6: Starting 2FA check loop (max 5 min)...")
        max_attempts = 60  # 5 seconds each → 5 minutes
        for attempt in range(max_attempts):
            try:
                heading = page.query_selector("#headingText")
                if heading and heading.text_content() == "2-Step Verification":
                    logger_config.info(f"[GoogleLogin] Step 6a: 2FA detected (attempt {attempt + 1}). Uploading screenshot...")
                    self._upload_screenshot(page, "2fa")
                    # Check "Don't ask again on this device" if present
                    checkbox = page.query_selector('input[type="checkbox"]')
                    if checkbox:
                        try:
                            checkbox.check()
                            logger_config.info("[GoogleLogin] Step 6b: Checked 'Don't ask again'.")
                        except Exception:
                            pass
                    page.wait_for_timeout(5000)
                else:
                    logger_config.info("[GoogleLogin] Step 6c: 2FA not detected, continuing.")
                    break
            except Exception as e:
                logger_config.error(f"[GoogleLogin] 2FA check error: {e}")
                break

    # ------------------------------------------------------------------
    # OAuth consent
    # ------------------------------------------------------------------

    def _handle_oauth_consent(self, page):
        """Handle brand-account selection and OAuth consent screens.

        Returns:
            True if auth code URL was successfully captured and saved
        """
        # Brand / channel account selection (YouTube-specific)
        for button in page.query_selector_all("button"):
            data_dest = button.get_attribute("data-destination-info")
            if data_dest and "Choosing an account will redirect you to" in data_dest:
                btn_text = button.inner_text()
                for account in page.query_selector_all("form li"):
                    if btn_text in account.inner_text():
                        clickable = account.query_selector("div")
                        if clickable:
                            logger_config.info(f"[GoogleLogin] Selecting brand account: {btn_text}")
                            clickable.focus()
                            page.wait_for_timeout(1000)
                            clickable.click(force=True)
                            page.wait_for_timeout(10000)
                        break
                break

        # Consent / Continue loop
        max_attempts = 10
        for i in range(max_attempts):
            logger_config.info(f"[GoogleLogin] Consent loop {i + 1}/{max_attempts}")
            page.wait_for_timeout(10000)
            self._upload_screenshot(page, f"consent_{i + 1}")

            # Consent page: checkbox + Continue
            form_checkbox = page.query_selector('form input[type="checkbox"]')
            if form_checkbox:
                logger_config.info("[GoogleLogin] Consent page detected.")
                if not form_checkbox.is_checked():
                    form_checkbox.click(force=True)
                    page.wait_for_timeout(1000)
                for button in page.query_selector_all("button"):
                    if button.inner_text() == "Continue":
                        logger_config.info("[GoogleLogin] Clicking final Continue on consent page.")
                        button.click(force=True)
                        page.wait_for_timeout(10000)
                        url = self._capture_url_from_address_bar()
                        if url and "code=" in url:
                            self._save_auth_code_url(url)
                            return True
                        self._upload_screenshot(page, f"consent_failed_{i + 1}")
                        return False

            # Intermediate page: Continue only
            for button in page.query_selector_all("button"):
                if button.inner_text() == "Continue":
                    logger_config.info("[GoogleLogin] Clicking Continue on intermediate page.")
                    button.click(force=True)
                    page.wait_for_timeout(5000)
                    url = self._capture_url_from_address_bar()
                    if url and "code=" in url:
                        self._save_auth_code_url(url)
                        return True
                    break  # not final, loop again

            # Check URL directly (redirect may have already happened)
            url = self._capture_url_from_address_bar()
            if url and "code=" in url:
                self._save_auth_code_url(url)
                return True

            logger_config.info("[GoogleLogin] No actionable consent elements found, retrying...")

        logger_config.warning("[GoogleLogin] OAuth consent did not complete within attempt limit.")
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_credentials(self):
        if self.email and self.password:
            return self.email, self.password
        if not self.email:
            self.email = input("Enter your Google email: ").strip()
        if not self.password:
            self.password = getpass.getpass("Enter your Google password (or app password): ")
        if not self.email or not self.password:
            raise ValueError("Email and password are required for Google login automation.")
        return self.email, self.password

    def _clear_cookies(self):
        if not self.browser_profile_path:
            return
        cookies_path = os.path.join(self.browser_profile_path, "Default", "Cookies")
        if os.path.exists(cookies_path):
            try:
                os.remove(cookies_path)
                logger_config.info(f"[GoogleLogin] Cleared session cookies: {cookies_path}")
            except Exception as e:
                logger_config.warning(f"[GoogleLogin] Could not clear cookies: {e}")

    def _upload_screenshot(self, page, label="screenshot"):
        _upload_screenshot_to_hf(page, self.hf_screenshot_repo_id, label)

    def _save_auth_code_url(self, url):
        if self.authorization_code_path:
            os.makedirs(os.path.dirname(self.authorization_code_path), exist_ok=True)
            with open(self.authorization_code_path, "w") as f:
                f.write(url)
            logger_config.info(f"[GoogleLogin] Auth code URL saved to {self.authorization_code_path}")

    def _capture_url_from_address_bar(self):
        """Get current browser URL via CDP, with xdotool+xclip Docker fallback.

        Returns:
            URL string or None
        """
        # Method 1: Chrome DevTools Protocol (preferred)
        try:
            cdp_url = f"http://localhost:{self._cdp_port}/json"
            with urllib.request.urlopen(cdp_url, timeout=5) as resp:
                pages = json.loads(resp.read().decode())
                if pages:
                    url = pages[0].get("url", "")
                    if url and not url.startswith("chrome://") and not url.startswith("chrome-error://"):
                        logger_config.info(f"[GoogleLogin] URL via CDP: {url}")
                        return url
        except Exception as e:
            logger_config.warning(f"[GoogleLogin] CDP failed: {e}")

        # Method 2: xdotool + xclip (Docker fallback)
        if not self.docker_name:
            return None
        try:
            for keys in ["ctrl+l", "ctrl+a", "ctrl+c"]:
                subprocess.run(
                    ["docker", "exec", self.docker_name, "xdotool", "key", keys],
                    timeout=5, check=True,
                )
                time.sleep(0.4)
            result = subprocess.run(
                ["docker", "exec", self.docker_name, "xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                url = result.stdout.strip()
                logger_config.info(f"[GoogleLogin] URL via xclip: {url}")
                return url
            logger_config.warning(f"[GoogleLogin] xclip failed: {result.stderr}")
        except Exception as e:
            logger_config.error(f"[GoogleLogin] xdotool/xclip error: {e}")

        return None
