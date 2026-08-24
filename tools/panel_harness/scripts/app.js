// Stub of ComfyUI's app module, enough to mount the panel.
window.LiteGraph = {
  createNode: (type) => ({ type, comfyClass: type, pos: [0, 0], widgets: [{ name: "manifest", value: "" }] }),
};

export const app = {
  graph: {
    _nodes: [],
    serialize: () => ({ nodes: [] }),
    setDirtyCanvas() {},
    add(node) {
      this._nodes.push(node);
    },
  },
  graphToPrompt: async () => ({ output: {} }),
  registerExtension(extension) {
    window.__extension = extension;
    extension.setup?.();
  },
  extensionManager: {
    // The panel reads the open workflow from here to scope its state.
    workflow: {
      get activeWorkflow() {
        return { path: window.__workflow ?? "workflows/one.json" };
      },
    },
    registerSidebarTab(tab) {
      window.__tab = tab;
    },
    setting: { get: () => undefined, set: async () => {} },
  },
};
