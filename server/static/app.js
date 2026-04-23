/**
 * Oráculo Kiaiá — chat local (jobs no servidor + polling; Screen Wake Lock durante geração)
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
  const sessionListEl = document.getElementById("session-list");
  const sessionListMenuEl = document.getElementById("session-list-menu");

  const history = [];
  let currentSessionId = null;
  let lastSessions = [];

  /** AbortController do pedido (polling) atual; Parar = abort. */
  let streamAborter = null;

  /** Geração em background no servidor; cancelamento explícito via /cancel. */
  let currentJobId = null;

  let screenWakeLock = null;

  /** Bloco cuja ação “copiar” fica visível até outro botão copiar ser clicado. */
  let copyPinnedStack = null;

  function setCopyPinnedStack(stack) {
    if (copyPinnedStack && copyPinnedStack !== stack) {
      copyPinnedStack.classList.remove("msg-stack--copy-pinned");
    }
    copyPinnedStack = stack || null;
    if (copyPinnedStack) {
      copyPinnedStack.classList.add("msg-stack--copy-pinned");
    }
  }

  function abortStream() {
    if (currentJobId) {
      const jid = currentJobId;
      currentJobId = null;
      apiFetch("/api/chat/jobs/" + encodeURIComponent(jid) + "/cancel", { method: "POST" }).catch(
        function () {}
      );
    }
    if (streamAborter) {
      streamAborter.abort();
      streamAborter = null;
    }
  }

  async function acquireScreenWakeLock() {
    try {
      if (!navigator.wakeLock || document.visibilityState !== "visible") return;
      if (screenWakeLock) return;
      screenWakeLock = await navigator.wakeLock.request("screen");
      screenWakeLock.addEventListener("release", function () {
        screenWakeLock = null;
      });
    } catch (_) {
      /* Permissão ou não suportado */
    }
  }

  function releaseScreenWakeLock() {
    if (screenWakeLock) {
      screenWakeLock.release().catch(function () {});
      screenWakeLock = null;
    }
  }

  /** Intervalo entre pedidos GET ao estado do job (geração em background no servidor). */
  const JOB_POLL_MS = 400;

  function sleepPoll(ms, signal) {
    return new Promise(function (resolve, reject) {
      if (signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      const t = setTimeout(resolve, ms);
      signal.addEventListener(
        "abort",
        function () {
          clearTimeout(t);
          reject(new DOMException("Aborted", "AbortError"));
        },
        { once: true }
      );
    });
  }

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

  /** Sessão por cookie: incluir credenciais em pedidos à API. */
  function apiFetch(path, init) {
    const o = init ? Object.assign({}, init) : {};
    o.credentials = o.credentials || "same-origin";
    return fetch(apiUrl(path), o);
  }

  const MAX_INPUT_HEIGHT = 320;
  const SWIPE_CLOSE_PX = 72;
  const MEDIA_MOBILE = window.matchMedia("(max-width: 768px)");

  function autoResizeInput() {
    inputEl.style.height = "auto";
    const h = Math.min(inputEl.scrollHeight, MAX_INPUT_HEIGHT);
    inputEl.style.height = h + "px";
  }

  const SEND_BTN_ICON =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V6"/><path d="m5 12 7-6 7 6"/></svg>';

  const STOP_BTN_ICON =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>';

  function setNewChatButtonsDisabled(disabled) {
    if (newChatBtn) {
      newChatBtn.disabled = disabled;
      newChatBtn.setAttribute("aria-disabled", disabled ? "true" : "false");
    }
    if (mobileNewChatBtn) {
      mobileNewChatBtn.disabled = disabled;
      mobileNewChatBtn.setAttribute("aria-disabled", disabled ? "true" : "false");
    }
    if (menuNewChatBtn) {
      menuNewChatBtn.disabled = disabled;
      menuNewChatBtn.setAttribute("aria-disabled", disabled ? "true" : "false");
    }
  }

  function updateSendState() {
    const busy = sendBtn.dataset.busy === "1";
    if (busy) {
      sendBtn.disabled = false;
      sendBtn.classList.add("send-btn--stop");
      sendBtn.setAttribute("aria-label", "Parar geração");
      sendBtn.setAttribute("title", "Parar");
      sendBtn.innerHTML = STOP_BTN_ICON;
      setNewChatButtonsDisabled(true);
    } else {
      sendBtn.classList.remove("send-btn--stop");
      sendBtn.setAttribute("aria-label", "Enviar mensagem");
      sendBtn.setAttribute("title", "Enviar");
      sendBtn.innerHTML = SEND_BTN_ICON;
      const hasText = inputEl.value.trim().length > 0;
      sendBtn.disabled = !hasText;
      setNewChatButtonsDisabled(false);
    }
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
      const r = await apiFetch("/api/status");
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      const j = await r.json();
      if (j.ui_only) {
        setStatusText("UI (sem modelo)", false);
      } else if (j.loaded) {
        setStatusText(j.mode + " · " + j.model_name, true);
      } else {
        setStatusText("A carregar modelo…", false);
      }
    } catch {
      setStatusText("Sem ligação", false);
    }
  }

  async function loadSessionUser() {
    try {
      const r = await apiFetch("/api/auth/me", { method: "GET" });
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      const j = await r.json();
      if (j && j.authenticated && j.username) {
        const desk = document.getElementById("user-name-desk");
        const menu = document.getElementById("user-name-menu");
        if (desk) {
          desk.textContent = j.username;
          desk.setAttribute("title", j.username);
        }
        if (menu) {
          menu.textContent = j.username;
          menu.setAttribute("title", j.username);
        }
      }
    } catch (_) {}
  }

  async function doLogout() {
    try {
      await apiFetch("/api/auth/logout", { method: "POST" });
    } catch (_) {}
    window.location.href = "/login";
  }

  document.getElementById("logout-desk")?.addEventListener("click", doLogout);
  document.getElementById("logout-menu")?.addEventListener("click", doLogout);

  /** Só aplica se o utilizador estiver perto do fim; ao subir para ler, o streaming deixa de puxar a vista. */
  const LOG_BOTTOM_THRESHOLD_PX = 96;
  let logUserFollowingBottom = true;

  function getScrollContainer() {
    return mainScrollEl || logEl;
  }

  function isLogNearBottom() {
    const el = getScrollContainer();
    if (!el) return true;
    const gap = el.scrollHeight - el.clientHeight - el.scrollTop;
    return gap < LOG_BOTTOM_THRESHOLD_PX;
  }

  function scrollLog() {
    const el = getScrollContainer();
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }

  function scrollLogIfFollowing() {
    if (logUserFollowingBottom) {
      scrollLog();
    }
  }

  (function attachLogScrollListener() {
    const el = getScrollContainer();
    if (!el) return;
    el.addEventListener(
      "scroll",
      function () {
        logUserFollowingBottom = isLogNearBottom();
      },
      { passive: true }
    );
  })();

  const COPY_ICON_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';

  const CHECK_ICON_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>';

  function copyTextToClipboard(text) {
    if (typeof navigator !== "undefined" && navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(() => copyTextViaExecCommand(text));
    }
    return copyTextViaExecCommand(text);
  }

  function copyTextViaExecCommand(text) {
    return new Promise((resolve, reject) => {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "readonly");
      ta.style.cssText =
        "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;border:0;padding:0;margin:0;pointer-events:none;";
      document.body.appendChild(ta);
      try {
        ta.focus();
        ta.select();
        ta.setSelectionRange(0, text.length);
        const ok = document.execCommand("copy");
        if (ok) resolve();
        else reject(new Error("execCommand"));
      } catch (err) {
        reject(err);
      } finally {
        if (ta.parentNode) {
          document.body.removeChild(ta);
        }
      }
    });
  }

  function bindCopyButton(btn, getText, stack) {
    btn.type = "button";
    btn.className = "msg__copy";
    btn.setAttribute("aria-label", "Copiar mensagem");
    btn.setAttribute("data-tooltip", "Copiar mensagem");
    btn.innerHTML = COPY_ICON_SVG;
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const t = getText();
      if (!t) return;
      setCopyPinnedStack(stack);
      copyTextToClipboard(t).then(
        () => {
          if (btn._restoreIconTimer) {
            window.clearTimeout(btn._restoreIconTimer);
            btn._restoreIconTimer = null;
          }
          btn.classList.add("msg__copy--pulse-check");
          btn.innerHTML = CHECK_ICON_SVG;
          btn.setAttribute("aria-label", "Copiado");
          btn._restoreIconTimer = window.setTimeout(() => {
            btn._restoreIconTimer = null;
            btn.classList.remove("msg__copy--pulse-check");
            btn.innerHTML = COPY_ICON_SVG;
            btn.setAttribute("aria-label", "Copiar mensagem");
          }, 1000);
        },
        () => {}
      );
    });
  }

  function appendMessage(role, content) {
    const stack = document.createElement("div");
    stack.className =
      "msg-stack " + (role === "assistant" ? "msg-stack--assistant" : "msg-stack--user");
    const bubble = document.createElement("div");
    bubble.className = "msg " + role;
    const textEl = document.createElement("span");
    textEl.className = "msg__text";
    textEl.textContent = content;
    const copyBtn = document.createElement("button");
    bindCopyButton(copyBtn, () => textEl.textContent, stack);
    bubble.appendChild(textEl);
    stack.appendChild(bubble);
    stack.appendChild(copyBtn);
    logInner.appendChild(stack);
    updateEmptyState();
    scrollLog();
  }

  function appendAssistantStreaming() {
    const stack = document.createElement("div");
    stack.className = "msg-stack msg-stack--assistant";
    const bubble = document.createElement("div");
    bubble.className = "msg assistant streaming";
    const textEl = document.createElement("span");
    textEl.className = "msg__text";
    const copyBtn = document.createElement("button");
    bindCopyButton(copyBtn, () => textEl.textContent, stack);
    bubble.appendChild(textEl);
    stack.appendChild(bubble);
    stack.appendChild(copyBtn);
    logInner.appendChild(stack);
    updateEmptyState();
    scrollLog();
    return { div: bubble, textEl };
  }

  function clearChat() {
    setCopyPinnedStack(null);
    history.length = 0;
    logInner.innerHTML = "";
    inputEl.value = "";
    autoResizeInput();
    delete sendBtn.dataset.busy;
    updateSendState();
    updateEmptyState();
    if (mobileMenu.classList.contains("is-open")) closeMobileMenu();
    inputEl.focus();
  }

  function buildSessionItem(s) {
    const li = document.createElement("li");
    li.className =
      "session-list__item" +
      (String(s.id) === String(currentSessionId) ? " session-list__item--active" : "");
    li.dataset.sid = String(s.id);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "session-list__btn";
    btn.textContent = s.title || "Conversa";
    li.appendChild(btn);
    return li;
  }

  function renderSessionList() {
    if (!sessionListEl || !sessionListMenuEl) return;
    sessionListEl.innerHTML = "";
    sessionListMenuEl.innerHTML = "";
    (lastSessions || []).forEach(function (s) {
      sessionListEl.appendChild(buildSessionItem(s));
      sessionListMenuEl.appendChild(buildSessionItem(s));
    });
  }

  async function loadSessionList() {
    const r = await apiFetch("/api/sessions");
    if (r.status === 401) return;
    if (!r.ok) return;
    const j = await r.json();
    lastSessions = j.sessions || [];
    renderSessionList();
  }

  async function openSession(sid, opts) {
    if (sendBtn.dataset.busy === "1") return;
    if (String(sid) === String(currentSessionId) && !(opts && opts.fromBoot)) {
      return;
    }
    const r = await apiFetch("/api/sessions/" + encodeURIComponent(sid));
    if (r.status === 401) {
      window.location.href = "/login";
      return;
    }
    if (!r.ok) {
      void loadSessionList();
      return;
    }
    const d = await r.json();
    currentSessionId = d.id;
    setCopyPinnedStack(null);
    history.length = 0;
    logInner.innerHTML = "";
    const msgs = d.messages || [];
    for (let i = 0; i < msgs.length; i++) {
      const m = msgs[i];
      if (m.role === "user" || m.role === "assistant") {
        history.push({ role: m.role, content: m.content });
        appendMessage(m.role, m.content);
      }
    }
    updateEmptyState();
    scrollLog();
    if (MEDIA_MOBILE.matches && mobileMenu && mobileMenu.classList.contains("is-open")) {
      closeMobileMenu();
    }
    void loadSessionList();
  }

  function onSessionListClick(e) {
    const btn = e.target.closest(".session-list__btn");
    if (!btn) return;
    const li = btn.closest("li");
    if (!li || li.dataset.sid == null) return;
    void openSession(li.dataset.sid, { force: true });
  }
  if (sessionListEl) {
    sessionListEl.addEventListener("click", onSessionListClick);
  }
  if (sessionListMenuEl) {
    sessionListMenuEl.addEventListener("click", onSessionListClick);
  }

  async function ensureSession() {
    if (currentSessionId != null) return true;
    const r = await apiFetch("/api/sessions", { method: "POST" });
    if (r.status === 401) {
      window.location.href = "/login";
      return false;
    }
    if (!r.ok) return false;
    const j = await r.json();
    currentSessionId = j.id;
    await loadSessionList();
    return true;
  }

  async function newChat() {
    if (sendBtn.dataset.busy === "1") return;
    const r = await apiFetch("/api/sessions", { method: "POST" });
    if (r.status === 401) {
      window.location.href = "/login";
      return;
    }
    if (!r.ok) return;
    const j = await r.json();
    currentSessionId = j.id;
    clearChat();
    await loadSessionList();
  }

  async function bootChats() {
    const r = await apiFetch("/api/sessions");
    if (r.status === 401) {
      window.location.href = "/login";
      return;
    }
    const j = await r.json();
    const sessions = j.sessions || [];
    if (sessions.length === 0) {
      const cr = await apiFetch("/api/sessions", { method: "POST" });
      if (cr.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (!cr.ok) return;
      const cj = await cr.json();
      currentSessionId = cj.id;
      lastSessions = [
        { id: cj.id, title: cj.title, created_at: "", updated_at: "" },
      ];
      renderSessionList();
      return;
    }
    lastSessions = sessions;
    currentSessionId = sessions[0].id;
    renderSessionList();
    await openSession(sessions[0].id, { fromBoot: true });
  }

  if (newChatBtn) {
    newChatBtn.addEventListener("click", function () {
      void newChat();
    });
  }
  if (mobileNewChatBtn) {
    mobileNewChatBtn.addEventListener("click", function () {
      void newChat();
    });
  }
  if (menuNewChatBtn) {
    menuNewChatBtn.addEventListener("click", function () {
      void newChat();
    });
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text) return;
    if (!(await ensureSession())) return;
    const ac = new AbortController();
    streamAborter = ac;
    const signal = ac.signal;

    logUserFollowingBottom = true;

    inputEl.value = "";
    autoResizeInput();
    updateSendState();

    appendMessage("user", text);
    history.push({ role: "user", content: text });

    sendBtn.dataset.busy = "1";
    updateSendState();

    const { div: assistantDiv, textEl } = appendAssistantStreaming();
    let lastText = "";
    let endReason = "none";

    try {
      const createRes = await apiFetch("/api/chat/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.slice(),
          max_new_tokens: 2048,
          temperature: 0.7,
          top_p: 0.9,
          session_id: currentSessionId,
        }),
      });
      if (createRes.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (!createRes.ok) {
        let detail = createRes.statusText;
        try {
          const errBody = await createRes.json();
          detail =
            typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
        } catch (_) {}
        throw new Error(detail);
      }
      const createJson = await createRes.json();
      const jobId = createJson.job_id;
      if (!jobId) {
        throw new Error("Resposta sem job_id");
      }
      currentJobId = jobId;
      await acquireScreenWakeLock();

      for (;;) {
        const stRes = await apiFetch("/api/chat/jobs/" + encodeURIComponent(jobId), { signal });
        if (stRes.status === 401) {
          window.location.href = "/login";
          return;
        }
        if (!stRes.ok) {
          let det = stRes.statusText;
          try {
            const jerr = await stRes.json();
            det = typeof jerr.detail === "string" ? jerr.detail : String(stRes.status);
          } catch (_) {}
          throw new Error(det);
        }
        const st = await stRes.json();
        lastText = st.text != null ? String(st.text) : "";
        textEl.textContent = lastText;
        scrollLogIfFollowing();
        if (st.status === "done") {
          endReason = "done";
          break;
        }
        if (st.status === "error") {
          throw new Error(st.error || "Erro no modelo");
        }
        if (st.status === "cancelled") {
          endReason = "cancelled";
          break;
        }
        await sleepPoll(JOB_POLL_MS, signal);
      }
    } catch (e) {
      const isAbort =
        e &&
        (e.name === "AbortError" ||
          (typeof e.message === "string" && /aborted|abort/i.test(e.message)));
      if (isAbort) {
        if (endReason === "none") {
          endReason = "abort";
        }
      } else {
        endReason = "error";
        textEl.textContent = "Erro: " + e.message;
      }
    } finally {
      if (endReason === "done") {
        history.push({ role: "assistant", content: lastText });
      } else if (endReason === "cancelled" || endReason === "abort") {
        if (lastText.trim().length) {
          history.push({ role: "assistant", content: lastText });
        } else {
          const stack = assistantDiv.closest && assistantDiv.closest(".msg-stack");
          if (stack) {
            stack.remove();
            updateEmptyState();
          }
        }
      }
      assistantDiv.classList.remove("streaming");
      currentJobId = null;
      releaseScreenWakeLock();
      if (streamAborter === ac) {
        streamAborter = null;
      }
      delete sendBtn.dataset.busy;
      updateSendState();
      void loadSessionList();
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", function onSendOrStop() {
    if (sendBtn.dataset.busy === "1") {
      abortStream();
      return;
    }
    send();
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (sendBtn.dataset.busy === "1") return;
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

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible" && sendBtn.dataset.busy === "1") {
      void acquireScreenWakeLock();
    }
  });

  autoResizeInput();
  updateSendState();
  updateEmptyState();
  void (async function initPage() {
    await loadSessionUser();
    await bootChats();
    refreshStatus();
  })();
  setInterval(refreshStatus, 8000);
})();
