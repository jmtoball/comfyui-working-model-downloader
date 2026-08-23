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
    this.items = [];
    this.folders = [];
    this.jobs = [];
    this.polling = null;
    this.build();
    this.loadFolders();
    this.loadConfig();
    this.refreshJobs();
    api.addEventListener("wmd.progress", (event) => this.onJobEvent(event.detail));
  }

  // -- construction ------------------------------------------------------

  build() {
    this.root.classList.add("wmd-panel");

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
      el("div", { className: "wmd-section" }, [el("h4", { textContent: "Downloads" }), this.jobList]),
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
      this.items = data.items.map((item) => ({ ...item, selected: item.resolved && !item.existing_path }));
      this.renderResults();
      const ready = this.items.filter((item) => item.resolved).length;
      const stuck = this.items.length - ready;
      this.say(
        `${ready} ready` + (stuck ? `, ${stuck} need a decision` : "") + `.`,
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
    const selectable = this.items.some((item) => item.selected);
    this.downloadButton.disabled = !selectable;
    this.pinButton.disabled = !this.items.some((item) => item.resolved);
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

    const row = el("div", { className: `wmd-item wmd-tier-${item.tier}` }, [
      el("div", { className: "wmd-item-head" }, [
        check,
        el("span", { className: "wmd-item-name", textContent: item.filename || item.source_url }),
        item.existing_path ? el("span", { className: "wmd-badge", textContent: "on disk" }) : null,
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
    const items = this.selectedEntries();
    if (!items.length) return;
    try {
      await call("/download", { body: { items } });
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
    const index = this.jobs.findIndex((existing) => existing.id === job.id);
    if (index >= 0) this.jobs[index] = job;
    else this.jobs.push(job);
    this.renderJobs();
    this.schedulePoll();
  }

  async refreshJobs() {
    try {
      this.jobs = (await call("/jobs")).jobs;
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
        new Panel(element);
      },
    });
  },
});
