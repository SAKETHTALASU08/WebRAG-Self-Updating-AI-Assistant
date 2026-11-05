// WebRAG Frontend Client Application
let currentSiteId = null;
let crawlPollInterval = null;

// DOM Elements
const siteListEl = document.getElementById("site-list");
const siteViewEl = document.getElementById("site-view");
const noSiteEl = document.getElementById("no-site-selected");
const ollamaBadge = document.getElementById("ollama-status-badge");
const modelTag = document.getElementById("model-tag");

// Current Site Elements
const siteNameEl = document.getElementById("current-site-name");
const siteUrlEl = document.getElementById("current-site-url");
const siteStatusEl = document.getElementById("current-site-status");
const statPagesEl = document.getElementById("stat-total-pages");
const statChunksEl = document.getElementById("stat-total-chunks");
const statLastCrawledEl = document.getElementById("stat-last-crawled");
const bannerEl = document.getElementById("crawl-progress-banner");
const bannerTitleEl = document.getElementById("crawl-banner-title");
const bannerUrlEl = document.getElementById("crawl-banner-url");
const bannerMetricsEl = document.getElementById("crawl-banner-metrics");

// Modals
const createModal = document.getElementById("create-modal");
const chunksModal = document.getElementById("chunks-modal");
const chunksContainer = document.getElementById("chunks-container");

// Chat
const chatMessagesEl = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const queryInput = document.getElementById("query-input");
const btnSend = document.getElementById("btn-send");

// ---------------- Initialize & Health ---------------- //

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    if (data.ollama && data.ollama.connected) {
      ollamaBadge.textContent = "Ollama Active";
      ollamaBadge.style.color = "var(--success)";
      if (modelTag) {
        modelTag.textContent = `Model: ${data.ollama.target_model}`;
      }
    } else {
      ollamaBadge.textContent = "Ollama Offline";
      ollamaBadge.style.color = "var(--danger)";
    }
  } catch (err) {
    ollamaBadge.textContent = "Backend Offline";
    ollamaBadge.style.color = "var(--danger)";
  }
}

// ---------------- Sites Management ---------------- //

async function loadSites() {
  try {
    const res = await fetch("/api/sites");
    const sites = await res.json();
    renderSiteList(sites);

    // If current site is selected, refresh its stats
    if (currentSiteId) {
      const match = sites.find((s) => s.id === currentSiteId);
      if (match) {
        updateSiteDetails(match);
      } else {
        selectSite(null);
      }
    }
  } catch (err) {
    console.error("Failed to load sites:", err);
  }
}

function renderSiteList(sites) {
  if (!sites || sites.length === 0) {
    siteListEl.innerHTML = '<div class="empty-hint" style="padding:1rem;color:var(--text-muted);font-size:0.8rem;">No websites added yet. Click "+ Add Website" to begin.</div>';
    return;
  }

  siteListEl.innerHTML = sites
    .map((s) => {
      const isActive = s.id === currentSiteId ? "active" : "";
      return `
        <div class="site-card ${isActive}" onclick="selectSite('${s.id}')">
          <div class="site-card-title">${escapeHtml(s.name || s.root_url)}</div>
          <div class="site-card-url">${escapeHtml(s.root_url)}</div>
          <div class="site-card-meta">
            <span class="site-card-tag">${s.total_pages} pages</span>
            <span class="site-card-tag">${s.total_chunks} chunks</span>
          </div>
        </div>
      `;
    })
    .join("");
}

async function selectSite(siteId) {
  currentSiteId = siteId;
  if (!siteId) {
    siteViewEl.classList.add("hidden");
    noSiteEl.classList.remove("hidden");
    if (crawlPollInterval) clearInterval(crawlPollInterval);
    return;
  }

  noSiteEl.classList.add("hidden");
  siteViewEl.classList.remove("hidden");

  // Re-render site list active highlight
  loadSites();

  try {
    const res = await fetch(`/api/sites/${siteId}`);
    if (!res.ok) throw new Error("Site not found");
    const site = await res.json();
    updateSiteDetails(site);
    startCrawlStatusPolling();
  } catch (err) {
    console.error("Error fetching site details:", err);
  }
}

function updateSiteDetails(site) {
  siteNameEl.textContent = site.name || site.root_url;
  siteUrlEl.textContent = site.root_url;
  siteUrlEl.href = site.root_url;

  siteStatusEl.textContent = site.status;
  siteStatusEl.className = "status-chip " + (site.status === "ready" ? "ready" : site.status === "crawling" ? "crawling" : "");

  statPagesEl.textContent = site.total_pages;
  statChunksEl.textContent = site.total_chunks;

  if (site.last_crawled_at) {
    const date = new Date(site.last_crawled_at);
    statLastCrawledEl.textContent = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } else {
    statLastCrawledEl.textContent = "Never";
  }
}

// ---------------- Incremental Crawl & Polling ---------------- //

async function triggerRecrawl() {
  if (!currentSiteId) return;
  try {
    const res = await fetch(`/api/sites/${currentSiteId}/crawl`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to trigger update");
    startCrawlStatusPolling();
  } catch (err) {
    alert("Error triggering incremental crawl: " + err.message);
  }
}

function startCrawlStatusPolling() {
  if (crawlPollInterval) clearInterval(crawlPollInterval);

  async function poll() {
    if (!currentSiteId) return;
    try {
      const res = await fetch(`/api/sites/${currentSiteId}/crawl/status`);
      if (!res.ok) return;
      const statusData = await res.json();

      if (statusData.is_crawling) {
        bannerEl.classList.remove("hidden");
        bannerTitleEl.textContent = "Incremental Crawling in Progress...";
        bannerUrlEl.textContent = statusData.current_url || "Processing pages...";
        bannerMetricsEl.textContent = `${statusData.pages_crawled} / ${statusData.pages_total} pages`;
        siteStatusEl.textContent = "crawling";
        siteStatusEl.className = "status-chip crawling";
      } else {
        bannerEl.classList.add("hidden");
        if (crawlPollInterval) clearInterval(crawlPollInterval);
        loadSites();
      }
    } catch (e) {
      console.warn("Poll status error:", e);
    }
  }

  poll();
  crawlPollInterval = setInterval(poll, 1500);
}

// ---------------- Delete Site ---------------- //

async function deleteCurrentSite() {
  if (!currentSiteId) return;
  if (!confirm("Are you sure you want to delete this website knowledge base? This action cannot be undone.")) return;

  try {
    const res = await fetch(`/api/sites/${currentSiteId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to delete");
    currentSiteId = null;
    selectSite(null);
    loadSites();
  } catch (err) {
    alert("Error deleting site: " + err.message);
  }
}

// ---------------- Inspect Chunks Modal ---------------- //

async function openChunksModal() {
  if (!currentSiteId) return;
  chunksModal.classList.remove("hidden");
  chunksContainer.innerHTML = '<div class="empty-hint">Loading indexed chunks...</div>';

  try {
    const res = await fetch(`/api/sites/${currentSiteId}/chunks?limit=50`);
    if (!res.ok) throw new Error("Failed to load chunks");
    const chunks = await res.json();

    if (chunks.length === 0) {
      chunksContainer.innerHTML = '<div class="empty-hint">No chunks indexed yet. Try crawling the site first.</div>';
      return;
    }

    chunksContainer.innerHTML = chunks
      .map((c) => {
        const pageBadge = c.page_number ? `<span>Page ${c.page_number}</span>` : "";
        return `
          <div class="chunk-card">
            <div class="chunk-header">
              <span>#${c.chunk_index} &bull; <a href="${escapeHtml(c.url)}" target="_blank" style="color:var(--accent);">${escapeHtml(c.url)}</a> ${pageBadge}</span>
              <span class="chunk-hash" title="SHA-256 chunk hash">SHA: ${escapeHtml(c.chunk_hash.substring(0, 12))}...</span>
            </div>
            <div class="chunk-text">${escapeHtml(c.text)}</div>
          </div>
        `;
      })
      .join("");
  } catch (err) {
    chunksContainer.innerHTML = `<div class="empty-hint" style="color:var(--danger);">Error: ${escapeHtml(err.message)}</div>`;
  }
}

// ---------------- Chat & Streaming RAG ---------------- //

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query || !currentSiteId) return;

  // Append User message
  appendUserMessage(query);
  queryInput.value = "";
  queryInput.style.height = "auto";

  // Create Assistant placeholder message
  const assistantMsgEl = appendAssistantMessage();
  const contentEl = assistantMsgEl.querySelector(".message-body");
  contentEl.textContent = "";

  const cursorSpan = document.createElement("span");
  cursorSpan.className = "banner-spinner";
  cursorSpan.style.display = "inline-block";
  cursorSpan.style.width = "12px";
  cursorSpan.style.height = "12px";
  contentEl.appendChild(cursorSpan);

  let citations = [];

  try {
    const response = await fetch(`/api/sites/${currentSiteId}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || "Error querying knowledge base");
    }

    // Read SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    cursorSpan.remove();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep remainder

      for (const block of lines) {
        if (!block.trim()) continue;
        const eventMatch = block.match(/event:\s*(.+)/);
        const dataMatch = block.match(/data:\s*(.+)/s);

        const event = eventMatch ? eventMatch[1].trim() : "message";
        const rawData = dataMatch ? dataMatch[1].trim() : "";

        if (event === "citations") {
          try {
            citations = JSON.parse(rawData);
          } catch (e) {}
        } else if (event === "token") {
          try {
            const parsed = JSON.parse(rawData);
            contentEl.textContent += parsed.token;
            chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
          } catch (e) {}
        }
      }
    }

    // Render citations block if citations exist
    if (citations && citations.length > 0) {
      const citationsContainer = document.createElement("div");
      citationsContainer.className = "citations-container";

      const heading = document.createElement("div");
      heading.className = "citations-heading";
      heading.textContent = `Sources Cited (${citations.length})`;
      citationsContainer.appendChild(heading);

      const list = document.createElement("div");
      list.className = "citations-list";

      citations.forEach((c, idx) => {
        const chip = document.createElement("a");
        chip.className = "citation-chip";
        chip.href = c.url;
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
        const pageText = c.page_number ? ` (p. ${c.page_number})` : "";
        chip.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
          </svg>
          [${idx + 1}] ${escapeHtml(c.title || c.url)}${pageText}
        `;
        chip.title = `Score: ${(c.similarity_score * 100).toFixed(1)}%\n\nExcerpt:\n${c.snippet}`;
        list.appendChild(chip);
      });

      citationsContainer.appendChild(list);
      assistantMsgEl.querySelector(".message-bubble").appendChild(citationsContainer);
    }
  } catch (err) {
    cursorSpan.remove();
    contentEl.textContent = "Error: " + err.message;
    contentEl.style.color = "var(--danger)";
  }

  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
});

function appendUserMessage(text) {
  // Hide welcome message if visible
  const welcome = chatMessagesEl.querySelector(".assistant-welcome");
  if (welcome) welcome.remove();

  const row = document.createElement("div");
  row.className = "message-row user";
  row.innerHTML = `
    <div class="message-avatar">U</div>
    <div class="message-bubble">
      <div class="message-body">${escapeHtml(text)}</div>
    </div>
  `;
  chatMessagesEl.appendChild(row);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function appendAssistantMessage() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.innerHTML = `
    <div class="message-avatar">AI</div>
    <div class="message-bubble">
      <div class="message-body"></div>
    </div>
  `;
  chatMessagesEl.appendChild(row);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  return row;
}

// Auto-expand textarea
queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";
});

// Shift+Enter newline vs Enter send
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit"));
  }
});

// ---------------- Create Site Form ---------------- //

document.getElementById("create-site-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const rootUrl = document.getElementById("input-root-url").value.trim();
  const name = document.getElementById("input-site-name").value.trim();
  const maxDepth = parseInt(document.getElementById("input-max-depth").value) || 3;
  const maxPages = parseInt(document.getElementById("input-max-pages").value) || 50;

  try {
    const res = await fetch("/api/sites", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        root_url: rootUrl,
        name: name || undefined,
        max_depth: maxDepth,
        max_pages: maxPages,
      }),
    });

    if (!res.ok) throw new Error("Failed to register website");
    const newSite = await res.json();
    createModal.classList.add("hidden");
    document.getElementById("create-site-form").reset();

    loadSites();
    selectSite(newSite.id);
  } catch (err) {
    alert("Error: " + err.message);
  }
});

// ---------------- Modal Toggles & Button Handlers ---------------- //

document.getElementById("btn-open-create-modal").addEventListener("click", () => {
  createModal.classList.remove("hidden");
  document.getElementById("input-root-url").focus();
});

document.getElementById("btn-empty-add").addEventListener("click", () => {
  createModal.classList.remove("hidden");
  document.getElementById("input-root-url").focus();
});

document.getElementById("btn-close-create-modal").addEventListener("click", () => {
  createModal.classList.add("hidden");
});

document.getElementById("btn-cancel-create").addEventListener("click", () => {
  createModal.classList.add("hidden");
});

document.getElementById("btn-recrawl").addEventListener("click", triggerRecrawl);
document.getElementById("btn-delete-site").addEventListener("click", deleteCurrentSite);
document.getElementById("btn-inspect-chunks").addEventListener("click", openChunksModal);
document.getElementById("btn-close-chunks-modal").addEventListener("click", () => {
  chunksModal.classList.add("hidden");
});

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ---------------- Boot ---------------- //
checkHealth();
setInterval(checkHealth, 15000);
loadSites();
