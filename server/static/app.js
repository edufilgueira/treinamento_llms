/**
 * Oráculo Kiaiá — chat local (streaming SSE)
 */
(function () {
  "use strict";

  const logEl = document.getElementById("log");
  const mainScrollEl = document.querySelector(".main-scroll");
  const logInner = document.getElementById("log-inner");
  const emptyState = document.getElementById("empty-state");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const statusEl = document.getElementById("status");
  const menuStatusEl = document.getElementById("menu-status");
  const newChatBtn = document.getElementById("new-chat");
  const mobileNewChatBtn = document.getElementById("mobile-new-chat");
  const menuNewChatBtn = document.getElementById("menu-new-chat");
  const menuToggle = document.getElementById("menu-toggle");
  const mobileMenu = document.getElementById("mobile-menu");
  const mobileMenuPanel = document.getElementById("mobile-menu-panel");

  const history = [];

  /**
   * Com a página em file://, fetch("/api/...") não aponta para o servidor.
   * Usa 127.0.0.1:8765 por defeito; opcional: localStorage "oraculo_api_origin" = "http://host:porta"
   */
  const API_ORIGIN = (function () {
    try {
      const s = localStorage.getItem("oraculo_api_origin");
      if (s) return String(s).replace(/\/$/, "");
    } catch (_) {}
    if (window.location.protocol === "file:") return "http://127.0.0.1:8765";
    return "";
  })();

  function apiUrl(path) {
    const p = path.startsWith("/") ? path : "/" + path;
    return API_ORIGIN + p;
  }

  const MAX_INPUT_HEIGHT = 320;
  const SWIPE_CLOSE_PX = 72;
  const MEDIA_MOBILE = window.matchMedia("(max-width: 768px)");

  function autoResizeInput() {
    inputEl.style.height = "auto";
    const h = Math.min(inputEl.scrollHeight, MAX_INPUT_HEIGHT);
    inputEl.style.height = h + "px";
  }

  function updateSendState() {
    const hasText = inputEl.value.trim().length > 0;
    sendBtn.disabled = !hasText || sendBtn.dataset.busy === "1";
  }

  function updateEmptyState() {
    const hasMessages = logInner.children.length > 0;
    emptyState.hidden = hasMessages;
  }

  function setStatusText(text, ready) {
    statusEl.textContent = text;
    menuStatusEl.textContent = text;
    if (ready) {
      statusEl.classList.add("ready");
      menuStatusEl.classList.add("ready");
    } else {
      statusEl.classList.remove("ready");
      menuStatusEl.classList.remove("ready");
    }
  }

  function openMobileMenu() {
    if (!MEDIA_MOBILE.matches) return;
    mobileMenu.classList.add("is-open");
    mobileMenu.setAttribute("aria-hidden", "false");
    menuToggle.setAttribute("aria-expanded", "true");
    document.body.classList.add("menu-open");
    menuNewChatBtn.focus();
  }

  function closeMobileMenu() {
    mobileMenuPanel.classList.remove("is-dragging");
    mobileMenuPanel.style.transform = "";
    mobileMenu.classList.remove("is-open");
    mobileMenu.setAttribute("aria-hidden", "true");
    menuToggle.setAttribute("aria-expanded", "false");
    document.body.classList.remove("menu-open");
    menuToggle.focus();
  }

  function toggleMobileMenu() {
    if (mobileMenu.classList.contains("is-open")) closeMobileMenu();
    else openMobileMenu();
  }

  inputEl.addEventListener("input", () => {
    autoResizeInput();
    updateSendState();
  });

  let touchStartX = 0;
  let touchLastX = 0;
  let dragOffset = 0;

  mobileMenuPanel.addEventListener(
    "touchstart",
    (e) => {
      if (!mobileMenu.classList.contains("is-open")) return;
      touchStartX = e.touches[0].clientX;
      touchLastX = touchStartX;
      dragOffset = 0;
      mobileMenuPanel.classList.add("is-dragging");
    },
    { passive: true }
  );

  mobileMenuPanel.addEventListener(
    "touchmove",
    (e) => {
      if (!mobileMenu.classList.contains("is-open")) return;
      const x = e.touches[0].clientX;
      const dx = x - touchStartX;
      touchLastX = x;
      if (dx < 0) {
        dragOffset = dx;
        mobileMenuPanel.style.transform = "translateX(" + dx + "px)";
        e.preventDefault();
      }
    },
    { passive: false }
  );

  mobileMenuPanel.addEventListener("touchend", () => {
    if (!mobileMenu.classList.contains("is-open")) return;
    mobileMenuPanel.classList.remove("is-dragging");
    const releaseDx = touchLastX - touchStartX;
    mobileMenuPanel.style.transform = "";
    if (dragOffset < -SWIPE_CLOSE_PX || releaseDx < -SWIPE_CLOSE_PX) {
      closeMobileMenu();
    }
    dragOffset = 0;
  });

  mobileMenuPanel.addEventListener("touchcancel", () => {
    if (!mobileMenu.classList.contains("is-open")) return;
    mobileMenuPanel.classList.remove("is-dragging");
    mobileMenuPanel.style.transform = "";
    dragOffset = 0;
  });

  menuToggle.addEventListener("click", toggleMobileMenu);

  async function refreshStatus() {
    try {
        const r = await fetch(apiUrl("/api/status"));
      const j = await r.json();
      if (j.loaded) {
        setStatusText(j.mode + " · " + j.model_name, true);
      } else {
        setStatusText("A carregar modelo…", false);
      }
    } catch {
      setStatusText("Sem ligação", false);
    }
  }

  function scrollLog() {
    const el = mainScrollEl || logEl;
    el.scrollTop = el.scrollHeight;
  }

  function appendMessage(role, content) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.textContent = content;
    logInner.appendChild(div);
    updateEmptyState();
    scrollLog();
  }

  function appendAssistantStreaming() {
    const div = document.createElement("div");
    div.className = "msg assistant streaming";
    const textEl = document.createElement("span");
    div.appendChild(textEl);
    logInner.appendChild(div);
    updateEmptyState();
    scrollLog();
    return { div, textEl };
  }

  function clearChat() {
    history.length = 0;
    logInner.innerHTML = "";
    inputEl.value = "";
    autoResizeInput();
    updateSendState();
    updateEmptyState();
    if (mobileMenu.classList.contains("is-open")) closeMobileMenu();
    inputEl.focus();
  }

  newChatBtn.addEventListener("click", clearChat);
  mobileNewChatBtn.addEventListener("click", clearChat);
  menuNewChatBtn.addEventListener("click", clearChat);

  async function send() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    autoResizeInput();
    updateSendState();

    appendMessage("user", text);
    history.push({ role: "user", content: text });

    sendBtn.dataset.busy = "1";
    updateSendState();

    const { div: assistantDiv, textEl } = appendAssistantStreaming();
    let fullReply = "";

    try {
        const r = await fetch(apiUrl("/api/chat/stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.slice(),
          max_new_tokens: 2048,
          temperature: 0.7,
          top_p: 0.9,
        }),
      });
      if (!r.ok) {
        let detail = r.statusText;
        try {
          const errBody = await r.json();
          detail =
            typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
        } catch (_) {}
        throw new Error(detail);
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let sep;
        while ((sep = buf.indexOf("\n\n")) !== -1) {
          const block = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          for (const line of block.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6).trim();
            if (payload === "[DONE]") continue;
            try {
              const j = JSON.parse(payload);
              if (j.delta) {
                fullReply += j.delta;
                textEl.textContent += j.delta;
              }
            } catch (_) {}
          }
        }
        scrollLog();
      }
      assistantDiv.classList.remove("streaming");
      history.push({ role: "assistant", content: fullReply });
    } catch (e) {
      textEl.textContent = "Erro: " + e.message;
      assistantDiv.classList.remove("streaming");
    } finally {
      delete sendBtn.dataset.busy;
      updateSendState();
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", send);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) send();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && mobileMenu.classList.contains("is-open")) {
      closeMobileMenu();
    }
  });

  MEDIA_MOBILE.addEventListener("change", (e) => {
    if (!e.matches && mobileMenu.classList.contains("is-open")) {
      closeMobileMenu();
    }
  });

  autoResizeInput();
  updateSendState();
  updateEmptyState();
  refreshStatus();
  setInterval(refreshStatus, 8000);
})();
