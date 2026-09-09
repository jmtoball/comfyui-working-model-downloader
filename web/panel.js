/**
 * The Model Downloader sidebar panel.
 *
 * This is where the work happens: scan the open workflow, resolve what it needs,
 * override anything the heuristics got wrong, download, and then pin the result
 * into the workflow so headless runs reproduce it. Plain DOM and fetch, no build
 * step -- a custom node should not need a bundler.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BASE = "/working_model_downloader";
const STYLESHEET = new URL("./panel.css", import.meta.url).href;
const NODE_TYPE = "WMD_ModelDownloader";
const ACTIVE_JOB_STATES = new Set(["queued", "downloading"]);
// Only what the graph actually asks for is ticked by default. Notes routinely
// document alternatives, optional extras and whole directories, and across a
// corpus of real workflows 62% of documented links were never referenced by the
// graph -- offering those is useful, downloading them unasked is not.
const WANTED_BY_DEFAULT = new Set(["required", "unknown"]);
// What the server considers terminal. `paused` is neither: it is waiting for you.
const FINISHED_JOB_STATES = new Set(["done", "present", "error", "cancelled"]);
const isFinished = (job) => FINISHED_JOB_STATES.has(job.status);

// ComfyUI auto-loads the .js files in WEB_DIRECTORY but not the stylesheet.
if (!document.querySelector(`link[href="${STYLESHEET}"]`)) {
  document.head.append(Object.assign(document.createElement("link"), { rel: "stylesheet", href: STYLESHEET }));
}

async function call(path, options = {}) {
  const response = await api.fetchApi(BASE + path, {
    method: options.body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

function el(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of [].concat(children)) {
    if (child) node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

function bytes(value) {
  if (!value) return "";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? size : size.toFixed(1)} ${units[unit]}`;
}

/**
 * Which workflow is open, as best the frontend will tell us.
 *
 * Downloads and resolutions belong to a workflow, not to the panel: without this
 * the queue from the last graph is still listed under the next one. Unsaved
 * workflows share a key, which is the best that can be done for something with no
 * identity yet.
 */
function workflowKey() {
  const active =
    app.extensionManager?.workflow?.activeWorkflow ?? app.workflowManager?.activeWorkflow;
  return String(active?.path || active?.key || active?.filename || "unsaved");
}

function workflowLabel(key) {
  return key === "unsaved" ? "unsaved workflow" : key.split("/").pop();
}

/** Per-workflow scratch state. Best-effort: private browsing may refuse it. */
const store = {
  key: (workflow) => `wmd.items.${workflow}`,
  read(workflow) {
    try {
      return JSON.parse(localStorage.getItem(this.key(workflow)) || "[]");
    } catch {
      return [];
    }
  },
  write(workflow, items) {
    try {
      if (items.length) localStorage.setItem(this.key(workflow), JSON.stringify(items));
      else localStorage.removeItem(this.key(workflow));
    } catch {
      /* storage unavailable; the panel still works, it just forgets */
    }
  },
};

/** The manifest already pinned into this workflow, as resolvable items. */
function pinnedItems() {
  const node = app.graph._nodes.find(
    (candidate) => candidate.comfyClass === NODE_TYPE || candidate.type === NODE_TYPE,
  );
  const widget = node?.widgets?.find((candidate) => candidate.name === "manifest");
  let entries = [];
  try {
    entries = JSON.parse(widget?.value || "{}").entries || [];
  } catch {
    return [];
  }
  return entries.map((entry) => ({
    url: entry.url,
    source_url: entry.url,
    provider: entry.provider || "direct",
    filename: entry.filename,
    folder: entry.folder,
    size: entry.size ?? null,
    sha256: entry.sha256 ?? null,
    tier: "manifest",
    need: "required",
    reason: "already pinned in this workflow",
    origin: entry.origin || "manifest",
    origin_node: "",
    existing_path: null,
    resolved: true,
    slot: entry.slot || null,
    candidates: [],
  }));
}

/** Identity for merging: a file is the thing at this path in this folder. */
function itemKey(item) {
  return `${item.folder || "?"}/${item.filename || item.url}`;
}

/** The workflow in both shapes: the UI graph carries the notes, the prompt the wiring. */
async function currentWorkflow() {
  const graph = app.graph.serialize();
  let prompt = null;
  try {
    prompt = (await app.graphToPrompt()).output;
  } catch (error) {
    console.warn("[wmd] could not build the prompt view of this graph", error);
  }
  return { workflow: graph, prompt };
}

function findOrCreateNode() {
  const existing = app.graph._nodes.find((node) => node.comfyClass === NODE_TYPE || node.type === NODE_TYPE);
  if (existing) return existing;
  const node = LiteGraph.createNode(NODE_TYPE);
  if (!node) throw new Error(`${NODE_TYPE} is not registered; restart ComfyUI`);
  node.pos = [40, 40];
  app.graph.add(node);
  return node;
}

class Panel {
  constructor(root) {
    this.root = root;
    this.workflow = workflowKey();
    this.items = store.read(this.workflow);
    this.folders = [];
    this.jobs = [];
    this.polling = null;
    this.build();
    this.loadFolders().then(() => this.renderResults());
    this.loadConfig();
    this.renderResults();
    this.refreshJobs();
    this.onProgress = (event) => this.onJobEvent(event.detail);
    api.addEventListener("wmd.progress", this.onProgress);
    // ComfyUI gives no reliable cross-version event for switching workflows, and
    // comparing one string every couple of seconds costs nothing.
    this.watch = setInterval(() => this.syncWorkflow(), 2000);
  }

  /**
   * Detach from the page.
   *
   * ComfyUI calls a sidebar tab's render() again whenever it remounts the tab,
   * so without this each remount leaves behind a timer still polling and a
   * socket listener still writing into a panel nobody can see.
   */
  destroy() {
    clearInterval(this.watch);
    clearTimeout(this.polling);
    api.removeEventListener?.("wmd.progress", this.onProgress);
  }

  /** Follow the user between workflows, carrying each one's state with it. */
  syncWorkflow() {
    const current = workflowKey();
    if (current === this.workflow) return;
    store.write(this.workflow, this.items);
    this.workflow = current;
    this.items = store.read(current);
    this.jobs = [];
    this.renderResults();
    this.refreshJobs();
    this.say(`Switched to ${workflowLabel(current)}.`);
  }

  remember() {
    store.write(this.workflow, this.items);
  }

  // -- construction ------------------------------------------------------

  build() {
    this.root.classList.add("wmd-panel");
    this.root.replaceChildren();

    this.status = el("div", { className: "wmd-status" });

    this.urls = el("textarea", {
      className: "wmd-urls",
      rows: 3,
      placeholder: "Paste HuggingFace or Civitai links, one per line",
      spellcheck: false,
    });

    this.allowSearch = el("input", { type: "checkbox", id: "wmd-search" });

    const scanButton = el("button", { className: "wmd-primary", textContent: "Scan workflow" });
    scanButton.onclick = () => this.resolve({ includeWorkflow: true });

    const addButton = el("button", { textContent: "Resolve links" });
    addButton.onclick = () => this.resolve({ includeWorkflow: false });

    this.results = el("div", { className: "wmd-results" });

    this.downloadButton = el("button", { className: "wmd-primary", textContent: "Download selected", disabled: true });
    this.downloadButton.onclick = () => this.download();

    this.pinButton = el("button", { textContent: "Save to workflow", disabled: true });
    this.pinButton.onclick = () => this.pin();

    this.jobList = el("div", { className: "wmd-jobs" });

    this.clearButton = el("button", { className: "wmd-link", textContent: "clear finished" });
    this.clearButton.onclick = async () => {
      await call("/jobs/clear", { body: { workflow_key: this.workflow } });
      this.refreshJobs();
    };

    this.root.append(
      el("div", { className: "wmd-section" }, [
        el("div", { className: "wmd-row" }, [scanButton, addButton]),
        this.urls,
        el("label", { className: "wmd-check" }, [
          this.allowSearch,
          " search HuggingFace and Civitai for anything the workflow does not document",
        ]),
        this.status,
      ]),
      el("div", { className: "wmd-section wmd-grow" }, [
        el("h4", { textContent: "Models" }),
        this.results,
        el("div", { className: "wmd-row" }, [this.downloadButton, this.pinButton]),
      ]),
      el("div", { className: "wmd-section wmd-downloads" }, [
        el("div", { className: "wmd-head" }, [
          el("h4", { textContent: "Downloads" }),
          this.clearButton,
        ]),
        this.jobList,
      ]),
      this.buildSettings(),
    );
  }

  buildSettings() {
    this.hfToken = el("input", { type: "password", placeholder: "not set", spellcheck: false });
    this.civitaiKey = el("input", { type: "password", placeholder: "not set", spellcheck: false });
    this.configNote = el("div", { className: "wmd-muted" });

    const save = el("button", { textContent: "Save keys" });
    save.onclick = async () => {
      const body = {};
      if (this.hfToken.value) body.hf_token = this.hfToken.value;
      if (this.civitaiKey.value) body.civitai_api_key = this.civitaiKey.value;
      try {
        this.showConfig(await call("/config", { body }));
        this.hfToken.value = "";
        this.civitaiKey.value = "";
        this.say("API keys saved.");
      } catch (error) {
        this.say(String(error), true);
      }
    };

    const clear = el("button", { textContent: "Clear" });
    clear.onclick = async () => {
      this.showConfig(await call("/config", { body: { hf_token: "", civitai_api_key: "" } }));
      this.say("Saved keys cleared; environment variables apply again.");
    };

    const details = el("details", { className: "wmd-section" });
    details.append(
      el("summary", { textContent: "API keys" }),
      el("label", {}, ["HuggingFace token", this.hfToken]),
      el("label", {}, ["Civitai API key", this.civitaiKey]),
      el("div", { className: "wmd-row" }, [save, clear]),
      this.configNote,
    );
    return details;
  }

  // -- state -------------------------------------------------------------

  say(message, isError = false) {
    this.status.textContent = message;
    this.status.classList.toggle("wmd-error", Boolean(isError));
  }

  async loadFolders() {
    try {
      this.folders = (await call("/folders")).folders.map((folder) => folder.key);
    } catch (error) {
      console.warn("[wmd] could not list model folders", error);
    }
  }

  async loadConfig() {
    try {
      this.showConfig(await call("/config"));
    } catch (error) {
      console.warn("[wmd] could not read the config", error);
    }
  }

  showConfig(state) {
    const describe = (set, hint, source) =>
      set ? `set ${hint} (from ${source === "user" ? "this panel" : source})` : "not set";
    this.configNote.textContent =
      `HuggingFace: ${describe(state.hf_token_set, state.hf_token_hint, state.hf_token_source)}. ` +
      `Civitai: ${describe(state.civitai_api_key_set, state.civitai_api_key_hint, state.civitai_api_key_source)}. ` +
      "Keys are stored outside the workflow and never saved into it.";
  }

  // -- resolving ---------------------------------------------------------

  async resolve({ includeWorkflow }) {
    this.say("Resolving…");
    try {
      const body = {
        urls: this.urls.value,
        allow_search: this.allowSearch.checked,
        overrides: this.items.map((item) => ({
          source_url: item.source_url,
          folder: item.folder,
          filename: item.filename,
        })),
      };
      if (includeWorkflow) Object.assign(body, await currentWorkflow());
      const data = await call("/resolve", { body });
      const fresh = data.items.map((item) => ({
        ...item,
        selected: WANTED_BY_DEFAULT.has(item.need) && item.resolved && !item.existing_path,
      }));
      // A fresh resolution wins, but anything already pinned into the workflow is
      // kept: a model that has finished downloading no longer shows up as missing,
      // and dropping it here would mean it could never be pinned.
      const merged = new Map();
      for (const item of [...pinnedItems(), ...this.items, ...fresh]) {
        merged.set(itemKey(item), { ...(merged.get(itemKey(item)) || {}), ...item });
      }
      this.items = [...merged.values()];
      this.remember();
      this.renderResults();
      const needed = this.items.filter((item) => item.selected).length;
      const optional = this.items.filter((item) => item.need === "optional").length;
      const present = this.items.filter((item) => item.existing_path).length;
      const stuck = this.items.filter((item) => !item.resolved).length;
      this.say(
        `${needed} to download` +
          (present ? `, ${present} on disk` : "") +
          (optional ? `, ${optional} the graph does not use` : "") +
          (stuck ? `, ${stuck} need a decision` : "") +
          ` \u00b7 ${workflowLabel(this.workflow)}`,
      );
    } catch (error) {
      this.say(String(error), true);
    }
  }

  renderResults() {
    this.results.replaceChildren();
    if (!this.items.length) {
      this.results.append(el("div", { className: "wmd-muted", textContent: "Nothing resolved yet." }));
    }
    this.items.forEach((item, index) => this.results.append(this.renderItem(item, index)));
    const chosen = this.items.filter((item) => item.selected);
    const total = chosen.reduce((sum, item) => sum + (item.size || 0), 0);
    this.downloadButton.textContent = total
      ? `Download ${chosen.length} (${bytes(total)})`
      : "Download selected";
    this.downloadButton.disabled = !chosen.length;
    this.pinButton.disabled = !this.items.some((item) => item.resolved);
    this.remember();
  }

  renderItem(item, index) {
    const check = el("input", { type: "checkbox", checked: Boolean(item.selected), disabled: !item.resolved });
    check.onchange = () => {
      item.selected = check.checked;
      this.renderResults();
    };

    const filename = el("input", { className: "wmd-name", value: item.filename || "", spellcheck: false });
    filename.onchange = () => {
      item.filename = filename.value;
    };

    const folder = el("select", { className: "wmd-folder" });
    const options = this.folders.includes(item.folder) || !item.folder ? this.folders : [item.folder, ...this.folders];
    folder.append(el("option", { value: "", textContent: "— choose a folder —" }));
    for (const key of options) {
      folder.append(el("option", { value: key, textContent: key, selected: key === item.folder }));
    }
    folder.onchange = () => {
      item.folder = folder.value;
      item.resolved = Boolean(item.folder && item.filename && item.url);
      item.selected = item.resolved;
      this.renderResults();
    };

    const meta = [item.provider, bytes(item.size), item.origin_node ? `note ${item.origin_node}` : item.origin]
      .filter(Boolean)
      .join(" · ");

    const remove = el("button", {
      className: "wmd-remove",
      textContent: "\u00d7",
      title: "Remove from this list (and from what gets pinned)",
    });
    remove.onclick = () => {
      this.items.splice(index, 1);
      this.renderResults();
    };

    const row = el("div", { className: `wmd-item wmd-tier-${item.tier}` }, [
      el("div", { className: "wmd-item-head" }, [
        check,
        el("span", { className: "wmd-item-name", textContent: item.filename || item.source_url }),
        item.existing_path ? el("span", { className: "wmd-badge", textContent: "on disk" }) : null,
        item.need === "optional"
          ? el("span", {
              className: "wmd-badge wmd-badge-optional",
              textContent: "not used here",
              title: "Documented in this workflow, but no loader in it asks for this file",
            })
          : null,
        remove,
      ]),
      el("div", { className: "wmd-muted", textContent: meta }),
      el("div", { className: "wmd-reason", textContent: item.error || item.reason }),
      el("div", { className: "wmd-row" }, [folder, filename]),
    ]);

    if (item.candidates?.length > 1) row.append(this.renderCandidates(item));
    if (!item.resolved && !item.candidates?.length) row.append(this.renderManualEntry(item, index));
    return row;
  }

  renderCandidates(item) {
    const list = el("div", { className: "wmd-candidates" }, [
      el("div", { className: "wmd-muted", textContent: "Pick a source:" }),
    ]);
    for (const candidate of item.candidates) {
      const pick = el("button", { className: "wmd-link", textContent: `${candidate.provider} — ${candidate.url}` });
      pick.onclick = () => {
        Object.assign(item, {
          url: candidate.url,
          size: candidate.size,
          sha256: candidate.sha256,
          provider: candidate.provider,
          resolved: Boolean(item.folder),
          selected: Boolean(item.folder),
          tier: "manual",
          reason: "you chose this source",
          candidates: [],
        });
        this.renderResults();
      };
      list.append(pick);
    }
    return list;
  }

  /** For a model nothing could resolve: paste a link, and we remember it next time. */
  renderManualEntry(item) {
    const input = el("input", { placeholder: "paste a download link for this file", spellcheck: false });
    const use = el("button", { textContent: "Use link" });
    use.onclick = async () => {
      if (!input.value.trim()) return;
      try {
        const data = await call("/resolve", { body: { urls: input.value } });
        const found = data.items[0];
        if (!found) throw new Error("that link did not resolve to a file");
        Object.assign(item, {
          url: found.url,
          provider: found.provider,
          size: found.size,
          sha256: found.sha256,
          filename: item.filename || found.filename,
          folder: item.folder || found.folder,
          tier: "manual",
          reason: "you provided this link",
        });
        item.resolved = Boolean(item.folder && item.filename);
        item.selected = item.resolved;
        if (item.resolved) {
          await call("/rules", { body: { pattern: item.filename, field: "filename", folder: item.folder, url: item.url } });
        }
        this.renderResults();
      } catch (error) {
        this.say(String(error), true);
      }
    };
    return el("div", { className: "wmd-row" }, [input, use]);
  }

  // -- acting ------------------------------------------------------------

  selectedEntries() {
    return this.items
      .filter((item) => item.selected && item.resolved)
      .map((item) => ({
        url: item.url,
        filename: item.filename,
        folder: item.folder,
        provider: item.provider,
        sha256: item.sha256,
        size: item.size,
        slot: item.slot ? { node: item.slot.node, input: item.slot.input } : null,
        origin: item.origin,
        tier: item.tier,
      }));
  }

  async download() {
    const chosen = this.items.filter((item) => item.selected && item.resolved);
    const items = this.selectedEntries();
    if (!items.length) return;
    try {
      await call("/download", { body: { items, workflow_key: this.workflow } });
      // Handed over to the queue. Leaving them ticked means the next Download --
      // after adding one more model -- submits them all over again, and a job
      // whose file is already there still takes up a row saying so.
      for (const item of chosen) item.selected = false;
      this.renderResults();
      this.say(`Queued ${items.length} download(s).`);
      this.refreshJobs();
    } catch (error) {
      this.say(String(error), true);
    }
  }

  async pin() {
    const items = this.items
      .filter((item) => item.resolved)
      .map((item) => ({
        url: item.url,
        filename: item.filename,
        folder: item.folder,
        provider: item.provider,
        sha256: item.sha256,
        size: item.size,
        slot: item.slot ? { node: item.slot.node, input: item.slot.input } : null,
        origin: item.origin,
        tier: item.tier,
      }));
    try {
      const { manifest } = await call("/manifest", { body: { items } });
      const node = findOrCreateNode();
      const widget = node.widgets?.find((candidate) => candidate.name === "manifest");
      if (!widget) throw new Error("that node has no manifest widget");
      widget.value = manifest;
      node.onResize?.(node.size);
      app.graph.setDirtyCanvas(true, true);
      this.say(`Pinned ${items.length} model(s) into the workflow. Save it to keep them.`);
    } catch (error) {
      this.say(String(error), true);
    }
  }

  // -- jobs --------------------------------------------------------------

  onJobEvent(job) {
    if (!job?.id) return;
    // Progress is broadcast to every client for every job, so a download running
    // under another workflow arrives here too. Without this the panel is cleared
    // on switching and then refilled by the next progress tick.
    if (!this.ownsJob(job)) return;
    const index = this.jobs.findIndex((existing) => existing.id === job.id);
    if (index >= 0) this.jobs[index] = job;
    else this.jobs.push(job);
    if (this.noteFinished(job)) this.renderResults();
    this.renderJobs();
    this.schedulePoll();
  }

  /** A job with no workflow came from a queued prompt and belongs to all of them. */
  ownsJob(job) {
    return !job.workflow || job.workflow === this.workflow;
  }

  /**
   * Reflect a finished download back onto its row.
   *
   * Without this the list still claims the model is missing until the next scan,
   * so it stays eligible to be ticked and queued a second time.
   */
  noteFinished(job) {
    if (job.status !== "done" && job.status !== "present") return false;
    const base = (name) => String(name || "").replace(/\\/g, "/").split("/").pop();
    const item = this.items.find(
      (candidate) =>
        candidate.folder === job.folder && base(candidate.filename) === base(job.filename),
    );
    if (!item || item.existing_path) return false;
    item.existing_path = job.dest || true;
    item.selected = false;
    return true;
  }

  async refreshJobs() {
    try {
      const query = `?workflow=${encodeURIComponent(this.workflow)}`;
      this.jobs = (await call(`/jobs${query}`)).jobs;
      if (this.jobs.map((job) => this.noteFinished(job)).some(Boolean)) this.renderResults();
      this.renderJobs();
      this.schedulePoll();
    } catch (error) {
      console.warn("[wmd] could not list jobs", error);
    }
  }

  /** Poll only while something is running; events cover the rest. */
  schedulePoll() {
    const busy = this.jobs.some((job) => ACTIVE_JOB_STATES.has(job.status));
    if (!busy) {
      clearTimeout(this.polling);
      this.polling = null;
      return;
    }
    if (this.polling) return;
    this.polling = setTimeout(() => {
      this.polling = null;
      this.refreshJobs();
    }, 1000);
  }

  renderJobs() {
    this.jobList.replaceChildren();
    this.clearButton.disabled = !this.jobs.some(isFinished);
    if (!this.jobs.length) {
      this.jobList.append(el("div", { className: "wmd-muted", textContent: "No downloads yet." }));
      return;
    }
    for (const job of this.jobs) {
      const share = job.total ? Math.min(100, (job.downloaded / job.total) * 100) : 0;
      const bar = el("div", { className: "wmd-bar" }, [
        el("div", { className: "wmd-bar-fill", style: `width:${share}%` }),
      ]);
      const detail = job.error
        ? job.error
        : `${job.status} · ${bytes(job.downloaded)}${job.total ? ` / ${bytes(job.total)}` : ""}` +
          (job.speed ? ` · ${bytes(job.speed)}/s` : "");
      const row = el("div", { className: `wmd-job wmd-job-${job.status}` }, [
        el("div", { className: "wmd-item-name", textContent: `${job.folder}/${job.filename}` }),
        bar,
        el("div", { className: "wmd-muted", textContent: detail }),
      ]);
      if (ACTIVE_JOB_STATES.has(job.status)) {
        const cancel = el("button", { className: "wmd-link", textContent: "cancel" });
        cancel.onclick = () => call(`/jobs/${job.id}/cancel`, { body: {} }).then(() => this.refreshJobs());
        const pause = el("button", { className: "wmd-link", textContent: "pause" });
        pause.onclick = () => call(`/jobs/${job.id}/pause`, { body: {} }).then(() => this.refreshJobs());
        row.append(el("div", { className: "wmd-row" }, [pause, cancel]));
      } else if (job.status === "paused" || job.status === "error") {
        const retry = el("button", { className: "wmd-link", textContent: "resume" });
        retry.onclick = () => call(`/jobs/${job.id}/resume`, { body: {} }).then(() => this.refreshJobs());
        row.append(retry);
      }
      this.jobList.append(row);
    }
  }
}

app.registerExtension({
  name: "wmd.panel",
  settings: [
    {
      id: "wmd.concurrency",
      name: "Model Downloader: concurrent downloads",
      type: "slider",
      defaultValue: 2,
      attrs: { min: 1, max: 8, step: 1 },
      onChange: (value) => {
        api.fetchApi(`${BASE}/config`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prefs: { max_concurrent_downloads: value } }),
        }).catch(() => {});
      },
    },
    {
      id: "wmd.include_nsfw",
      name: "Model Downloader: include NSFW results when searching Civitai",
      type: "boolean",
      defaultValue: false,
      onChange: (value) => {
        api.fetchApi(`${BASE}/config`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prefs: { include_nsfw: Boolean(value) } }),
        }).catch(() => {});
      },
    },
  ],
  setup() {
    app.extensionManager.registerSidebarTab({
      id: "wmd",
      icon: "pi pi-download",
      title: "Model Downloader",
      tooltip: "Find and download the models this workflow needs",
      type: "custom",
      render: (element) => {
        // render() runs again on every remount. Replace the previous panel
        // rather than stacking a second copy of the whole UI on top of it.
        element.__wmdPanel?.destroy();
        element.__wmdPanel = new Panel(element);
      },
    });
  },
});
