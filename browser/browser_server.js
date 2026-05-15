const fs = require("fs");
const path = require("path");
const childProcess = require("child_process");

const rootDir = path.resolve(__dirname, "..");
const logsDir = path.join(rootDir, "logs", "browser");
const statePath = path.join(__dirname, "browser_state.json");
const latestScreenshotPath = path.join(logsDir, "latest.png");
const DEFAULT_BROWSER_PATHS = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
];

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function getPuppeteer() {
  try {
    return require("puppeteer-core");
  } catch (_error) {
    throw new Error(
      "puppeteer-core is not installed. Run `npm install --prefix browser` before using browser tools."
    );
  }
}

function readJsonStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      try {
        resolve(JSON.parse(data || "{}"));
      } catch (error) {
        reject(new Error(`Invalid JSON input: ${error.message}`));
      }
    });
    process.stdin.on("error", reject);
  });
}

function readState() {
  try {
    return JSON.parse(fs.readFileSync(statePath, "utf8"));
  } catch (_error) {
    return null;
  }
}

function writeState(state) {
  fs.writeFileSync(statePath, JSON.stringify(state, null, 2), "utf8");
}

function clearState() {
  try {
    fs.unlinkSync(statePath);
  } catch (_error) {
    return;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function discoverBrowserExecutable(env = process.env, candidatePaths = DEFAULT_BROWSER_PATHS) {
  const checkedPaths = [];
  const envPath = env.LOCALPILOT_BROWSER_EXECUTABLE;
  if (envPath) {
    checkedPaths.push(envPath);
    if (fs.existsSync(envPath)) {
      return {
        ok: true,
        path: envPath,
        source: "LOCALPILOT_BROWSER_EXECUTABLE",
        checked_paths: checkedPaths
      };
    }
  }

  for (const candidate of candidatePaths) {
    checkedPaths.push(candidate);
    if (fs.existsSync(candidate)) {
      return {
        ok: true,
        path: candidate,
        source: "common_windows_path",
        checked_paths: checkedPaths
      };
    }
  }

  return {
    ok: false,
    error: [
      "No supported browser executable was found.",
      "Set LOCALPILOT_BROWSER_EXECUTABLE to a valid Chrome or Edge path.",
      "Example: C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    ].join(" "),
    checked_paths: checkedPaths
  };
}

function buildBrowserExecutableError(action, discovery) {
  return {
    ok: false,
    action,
    error: discovery.error,
    checked_paths: discovery.checked_paths || []
  };
}

async function connectFromState() {
  const puppeteer = getPuppeteer();
  const state = readState();
  if (!state || (!state.wsEndpoint && !state.browserURL)) {
    return { ok: false, error: "No browser page is active." };
  }

  try {
    const browser = state.browserURL
      ? await puppeteer.connect({ browserURL: state.browserURL })
      : await puppeteer.connect({ browserWSEndpoint: state.wsEndpoint });
    let pages = await browser.pages();
    let page = pages[0];
    if (!page) {
      page = await browser.newPage();
    }
    return { ok: true, browser, page, state };
  } catch (error) {
    clearState();
    return { ok: false, error: `Browser session is unavailable: ${error.message}` };
  }
}

async function connectToBrowserURL(browserURL, timeoutMs = 20000) {
  const puppeteer = getPuppeteer();
  const deadline = Date.now() + timeoutMs;
  let lastError = "Browser did not open a debugging endpoint.";
  while (Date.now() < deadline) {
    try {
      const browser = await puppeteer.connect({ browserURL });
      let pages = await browser.pages();
      let page = pages[0];
      if (!page) {
        page = await browser.newPage();
      }
      return { ok: true, browser, page };
    } catch (error) {
      lastError = error.message;
      await sleep(500);
    }
  }
  return { ok: false, error: `Timed out waiting for browser debugging endpoint: ${lastError}` };
}

function chooseDebugPort() {
  return 9222 + Math.floor(Math.random() * 1000);
}

function timestamp() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate())
  ].join("") + "_" + [pad(now.getHours()), pad(now.getMinutes()), pad(now.getSeconds())].join("");
}

async function capturePageScreenshot(page) {
  ensureDir(logsDir);
  const filePath = path.join(logsDir, `browser_${timestamp()}.png`);
  await page.screenshot({ path: filePath, fullPage: true });
  fs.copyFileSync(filePath, latestScreenshotPath);
  return filePath;
}

async function getPageInfo(page, action, extra = {}) {
  const title = await page.title();
  const url = page.url();
  const text = await page.evaluate(() => {
    return document.body ? (document.body.innerText || "").replace(/\s+/g, " ").trim() : "";
  });
  const screenshotPath = await capturePageScreenshot(page);
  return {
    ok: true,
    action,
    url,
    title,
    text_preview: text.slice(0, 500),
    screenshot_path: screenshotPath,
    ...extra
  };
}

async function launchBrowser(payload) {
  ensureDir(logsDir);
  const discovery = discoverBrowserExecutable();
  if (!discovery.ok) {
    return buildBrowserExecutableError("launch_browser", discovery);
  }
  const debugPort = chooseDebugPort();
  const browserURL = `http://127.0.0.1:${debugPort}`;
  const userDataDir = path.join(logsDir, "profile");
  ensureDir(userDataDir);
  const args = [
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${userDataDir}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--new-window",
    "about:blank"
  ];
  if (payload.headless === true) {
    args.unshift("--headless=new");
  }
  const browserProcess = childProcess.spawn(discovery.path, args, {
    detached: true,
    stdio: "ignore"
  });
  browserProcess.unref();
  const connection = await connectToBrowserURL(browserURL);
  if (!connection.ok) {
    return {
      ok: false,
      action: "launch_browser",
      error: connection.error,
      browser_executable: discovery.path,
      browser_url: browserURL
    };
  }
  const { browser, page } = connection;
  writeState({
    wsEndpoint: browser.wsEndpoint(),
    browserURL,
    launchedAt: new Date().toISOString(),
    executablePath: discovery.path,
    browserPid: browserProcess ? browserProcess.pid : null
  });
  const result = await getPageInfo(page, "launch_browser", {
    browser_executable: discovery.path,
    browser_source: discovery.source,
    browser_pid: browserProcess ? browserProcess.pid : null,
    browser_url: browserURL
  });
  await browser.disconnect();
  return result;
}

async function closeBrowser() {
  const connection = await connectFromState();
  if (!connection.ok) {
    const state = readState();
    if (state && state.browserPid) {
      try {
        process.kill(state.browserPid);
        clearState();
        return { ok: true, action: "close_browser", closed_by_pid: state.browserPid };
      } catch (_error) {
        return { ok: false, action: "close_browser", error: connection.error };
      }
    }
    return { ok: false, action: "close_browser", error: connection.error };
  }
  const { browser } = connection;
  await browser.close();
  clearState();
  return { ok: true, action: "close_browser" };
}

async function gotoUrl(payload) {
  const connection = await connectFromState();
  if (!connection.ok) {
    return { ok: false, action: "goto_url", error: connection.error };
  }
  const { browser, page } = connection;
  try {
    await page.goto(payload.url, { waitUntil: "domcontentloaded", timeout: payload.timeout_ms || 30000 });
    return await getPageInfo(page, "goto_url");
  } finally {
    await browser.disconnect();
  }
}

async function searchWeb(payload) {
  const query = encodeURIComponent(payload.query || "");
  const url = payload.engine === "bing"
    ? `https://www.bing.com/search?q=${query}`
    : `https://www.google.com/search?q=${query}`;
  return gotoUrl({ url, timeout_ms: payload.timeout_ms });
}

async function clickSelector(payload) {
  const connection = await connectFromState();
  if (!connection.ok) {
    return { ok: false, action: "click_selector", error: connection.error };
  }
  const { browser, page } = connection;
  try {
    await page.waitForSelector(payload.selector, { timeout: payload.timeout_ms || 15000 });
    await page.click(payload.selector);
    return await getPageInfo(page, "click_selector", { selector: payload.selector });
  } finally {
    await browser.disconnect();
  }
}

async function typeSelector(payload) {
  const connection = await connectFromState();
  if (!connection.ok) {
    return { ok: false, action: "type_selector", error: connection.error };
  }
  const { browser, page } = connection;
  try {
    await page.waitForSelector(payload.selector, { timeout: payload.timeout_ms || 15000 });
    await page.click(payload.selector, { clickCount: 3 });
    await page.type(payload.selector, payload.text || "", { delay: payload.delay_ms || 20 });
    return await getPageInfo(page, "type_selector", { selector: payload.selector });
  } finally {
    await browser.disconnect();
  }
}

async function pressKey(payload) {
  const connection = await connectFromState();
  if (!connection.ok) {
    return { ok: false, action: "press_key", error: connection.error };
  }
  const { browser, page } = connection;
  try {
    await page.keyboard.press(payload.key);
    return await getPageInfo(page, "press_key", { key: payload.key });
  } finally {
    await browser.disconnect();
  }
}

async function getPageText() {
  const connection = await connectFromState();
  if (!connection.ok) {
    return { ok: false, action: "get_page_text", error: connection.error };
  }
  const { browser, page } = connection;
  try {
    const info = await getPageInfo(page, "get_page_text");
    return { ...info, text: info.text_preview };
  } finally {
    await browser.disconnect();
  }
}

async function screenshotPage() {
  const connection = await connectFromState();
  if (!connection.ok) {
    return { ok: false, action: "screenshot", error: connection.error };
  }
  const { browser, page } = connection;
  try {
    return await getPageInfo(page, "screenshot");
  } finally {
    await browser.disconnect();
  }
}

async function getPageInfoCommand() {
  const connection = await connectFromState();
  if (!connection.ok) {
    return { ok: false, action: "get_page_info", error: connection.error };
  }
  const { browser, page } = connection;
  try {
    return await getPageInfo(page, "get_page_info");
  } finally {
    await browser.disconnect();
  }
}

const handlers = {
  launch_browser: launchBrowser,
  close_browser: closeBrowser,
  goto_url: gotoUrl,
  search_web: searchWeb,
  click_selector: clickSelector,
  type_selector: typeSelector,
  press_key: pressKey,
  get_page_text: getPageText,
  screenshot: screenshotPage,
  get_page_info: getPageInfoCommand
};

async function main() {
  try {
    const payload = await readJsonStdin();
    const action = payload.action;
    if (!handlers[action]) {
      process.stdout.write(JSON.stringify({ ok: false, action, error: `Unknown browser action: ${action}` }));
      process.exit(1);
      return;
    }
    const result = await handlers[action](payload);
    process.stdout.write(JSON.stringify(result));
    process.exit(result.ok ? 0 : 1);
  } catch (error) {
    process.stdout.write(JSON.stringify({ ok: false, error: error.message }));
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = {
  DEFAULT_BROWSER_PATHS,
  discoverBrowserExecutable,
  buildBrowserExecutableError
};
