/**
 * Oráculo Kiaiá — chat local (jobs no servidor + polling; Screen Wake Lock durante geração)
 */
(function () {
  "use strict";

  const logEl = document.getElementById("log");
  const mainScrollEl = document.querySelector(".main-scroll");
  const logInner = document.getElementById("log-inner");
  const serverBusyBanner = document.getElementById("server-busy-banner");
  const composerOuter = document.getElementById("composer-outer");
  const emptyState = document.getElementById("empty-state");
  const inputEl = document.getElementById("input");
  const composerEl = document.getElementById("composer");
  const composerPlusBtn = document.getElementById("composer-plus");
  const sendBtn = document.getElementById("send");
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
  /** Sessão cujo título está a ser editado na barra (null = nenhum). */
  let editingSessionId = null;
  let savingSessionRename = false;

  const SESSION_LIST_LABEL_MAX = 30;

  function formatSessionListLabel(title) {
    const raw = String(title == null || String(title).trim() === "" ? "Conversa" : title).trim();
    if (raw.length <= SESSION_LIST_LABEL_MAX) {
      return raw;
    }
    return raw.slice(0, SESSION_LIST_LABEL_MAX) + "...";
  }

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
  const GEN_STATUS_POLL_MS = 2500;
  let generationBlockedByOther = false;
  let genStatusTimer = null;

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
   * Em file:// usa localhost:porta por defeito; opcional: localStorage "oraculo_api_origin".
   */
  const API_ORIGIN = (function () {
    try {
      const s = localStorage.getItem("oraculo_api_origin");
      if (s) return String(s).replace(/\/$/, "");
    } catch (_) {}
    if (window.location.protocol === "file:") return "http://localhost:8765";
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

  if (typeof marked !== "undefined" && typeof marked.setOptions === "function") {
    marked.setOptions({ gfm: true, breaks: true, mangle: false, headerIds: false });
  }

  function getMsgCopyText(textEl) {
    const d = textEl.getAttribute("data-raw-md");
    if (d != null) {
      return d;
    }
    return textEl.textContent || "";
  }

  let dompurifyLinksHooked = false;

  function renderMsgMarkdown(textEl, raw) {
    textEl.classList.add("msg__text--md");
    const md = raw == null ? "" : String(raw);
    textEl.setAttribute("data-raw-md", md);
    if (md.length === 0) {
      textEl.innerHTML = "";
      return;
    }
    if (typeof marked === "undefined" || typeof marked.parse !== "function") {
      textEl.textContent = md;
      return;
    }
    if (typeof DOMPurify === "undefined" || typeof DOMPurify.sanitize !== "function") {
      textEl.textContent = md;
      return;
    }
    if (!dompurifyLinksHooked && typeof DOMPurify.addHook === "function") {
      dompurifyLinksHooked = true;
      DOMPurify.addHook("afterSanitizeAttributes", function (node) {
        if (node.tagName === "A" && node.hasAttribute("href")) {
          node.setAttribute("target", "_blank");
          node.setAttribute("rel", "noopener noreferrer");
        }
      });
    }
    try {
      const html = marked.parse(md);
      textEl.innerHTML = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    } catch (_e) {
      textEl.textContent = md;
    }
  }

  const SWIPE_CLOSE_PX = 72;
  const MEDIA_MOBILE = window.matchMedia("(max-width: 768px)");
  /** Coarse/touch UA: usar tap na bolha para mostrar Copiar (não há hover). */
  const MEDIA_HOVER_NONE =
    typeof window.matchMedia === "function" ? window.matchMedia("(hover: none)") : null;

  function prefersNoHover() {
    return MEDIA_HOVER_NONE && MEDIA_HOVER_NONE.matches;
  }

  /** Clic fora das mensagens: esconder botão Copiar destacado em telemóvel. */
  let copyPinnedOutsideCloserInstalled = false;
  function ensureCopyPinnedOutsideCloser() {
    if (copyPinnedOutsideCloserInstalled) return;
    copyPinnedOutsideCloserInstalled = true;
    document.addEventListener(
      "click",
      function (ev) {
        if (!prefersNoHover()) return;
        var t = ev.target;
        if (!t.closest || !t.closest(".msg-stack")) {
          setCopyPinnedStack(null);
        }
      },
      false
    );
  }

  function bindMessageTapToRevealCopy(stack, bubbleEl) {
    ensureCopyPinnedOutsideCloser();
    bubbleEl.addEventListener("click", function (ev) {
      if (!prefersNoHover()) return;
      var el = ev.target;
      if (el.closest && (el.closest("a") || el.closest("button"))) return;
      setCopyPinnedStack(stack);
    });
  }

  function updateComposerExpanded() {
    if (!composerEl || !inputEl) return;
    const style = getComputedStyle(inputEl);
    const lh = parseFloat(style.lineHeight);
    const lineH = Number.isFinite(lh) && lh > 0 ? lh : parseFloat(style.fontSize || "16") * 1.45;
    const hasBreak = inputEl.value.indexOf("\n") !== -1;
    const multiLine = hasBreak || inputEl.scrollHeight > Math.ceil(lineH) + 8;
    composerEl.classList.toggle("composer--expanded", multiLine);
  }

  function autoResizeInput() {
    inputEl.style.height = "auto";
    /* Altura = conteúdo; o scroll fica no .composer-body (estilo ChatGPT) */
    inputEl.style.height = inputEl.scrollHeight + "px";
    updateComposerExpanded();
    const body = inputEl.closest(".composer-body");
    if (body) {
      const gap = body.scrollHeight - body.clientHeight - body.scrollTop;
      if (gap < 56) {
        body.scrollTop = body.scrollHeight;
      }
    }
  }

  /** Uma linha, sem composer--expanded — após enviar (imediatamente) e depois da resposta. */
  function collapseComposerToSingleRow() {
    if (!composerEl || !inputEl) return;
    composerEl.classList.remove("composer--expanded");
    inputEl.style.removeProperty("height");
    autoResizeInput();
    const body = inputEl.closest(".composer-body");
    if (body) {
      body.scrollTop = 0;
    }
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
      const blockSend = generationBlockedByOther;
      sendBtn.disabled = !hasText || blockSend;
      setNewChatButtonsDisabled(false);
    }
  }

  function applyGenerationStatus(active, yours) {
    const other = active && !yours;
    generationBlockedByOther = other;
    if (serverBusyBanner) {
      serverBusyBanner.hidden = !other;
    }
    if (composerOuter) {
      composerOuter.classList.toggle("composer-outer--server-busy", other);
    }
    updateSendState();
  }

  async function tickGenerationStatus() {
    try {
      const r = await apiFetch("/api/chat/generation-status");
      if (r.status === 401) {
        return;
      }
      if (!r.ok) {
        return;
      }
      const j = await r.json();
      applyGenerationStatus(!!j.active, !!j.yours);
    } catch (_) {}
  }

  function startGenerationStatusPolling() {
    if (genStatusTimer) {
      return;
    }
    void tickGenerationStatus();
    genStatusTimer = setInterval(tickGenerationStatus, GEN_STATUS_POLL_MS);
  }

  function updateEmptyState() {
    const hasMessages = logInner.children.length > 0;
    emptyState.hidden = hasMessages;
  }

  function openMobileMenu() {
    if (!MEDIA_MOBILE.matches) return;
    if (!mobileMenu || !menuToggle || !menuNewChatBtn) return;
    mobileMenu.classList.add("is-open");
    mobileMenu.setAttribute("aria-hidden", "false");
    menuToggle.setAttribute("aria-expanded", "true");
    document.body.classList.add("menu-open");
    menuNewChatBtn.focus();
  }

  function closeMobileMenu() {
    if (mobileMenuPanel) {
      mobileMenuPanel.classList.remove("is-dragging");
      mobileMenuPanel.style.transform = "";
    }
    if (mobileMenu) {
      mobileMenu.classList.remove("is-open");
      mobileMenu.setAttribute("aria-hidden", "true");
    }
    if (menuToggle) {
      menuToggle.setAttribute("aria-expanded", "false");
      menuToggle.focus();
    }
    document.body.classList.remove("menu-open");
  }

  function closeMobileMenuIfOpen() {
    if (MEDIA_MOBILE.matches && mobileMenu && mobileMenu.classList.contains("is-open")) {
      closeMobileMenu();
    }
  }

  function toggleMobileMenu() {
    if (!mobileMenu) return;
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

  const accountMenuDesk = document.getElementById("account-menu-desk");
  const accountTriggerDesk = document.getElementById("account-trigger-desk");
  const accountDdDesk = document.getElementById("account-dd-desk");
  const modalBackdrop = document.getElementById("modal-backdrop");
  const modalProfile = document.getElementById("modal-profile");
  const modalSettings = document.getElementById("modal-settings");
  const profileDisplayInput = document.getElementById("profile-display-input");
  const profileEmailInput = document.getElementById("profile-email-input");
  const profileLoginLine = document.getElementById("profile-login-line");
  const settingsModalHint = document.getElementById("settings-modal-hint");
  const settingsBlockGlobal = document.getElementById("settings-block-global");
  const settingsGlobalSystemPrompt = document.getElementById("settings-global-system-prompt");
  const settingsSystemPrompt = document.getElementById("settings-system-prompt");
  const settingsBlockLlama = document.getElementById("settings-block-llama");
  const settingsLlamaEnabled = document.getElementById("settings-llama-enabled");
  const settingsLlamaHost = document.getElementById("settings-llama-host");
  const settingsLlamaPort = document.getElementById("settings-llama-port");
  const settingsLlamaNCtx = document.getElementById("settings-llama-n-ctx");
  const settingsLlamaMaxTokens = document.getElementById("settings-llama-max-tokens");
  const settingsLlamaTemp = document.getElementById("settings-llama-temp");
  const settingsLlamaTopP = document.getElementById("settings-llama-top-p");
  const settingsLlamaRepeatPenalty = document.getElementById("settings-llama-repeat-penalty");
  const settingsLlamaRepeatLastN = document.getElementById("settings-llama-repeat-last-n");
  const settingsLlamaReasoning = document.getElementById("settings-llama-reasoning");
  const settingsLlamaReasoningBudgetOn = document.getElementById("settings-llama-reasoning-budget-on");
  let currentUserIsAdmin = false;

  function isAdminFromApi(v) {
    return v === true || v === 1;
  }

  function setAdminNavVisible(show) {
    document.querySelectorAll(".admin-only").forEach(function (el) {
      el.hidden = !show;
    });
  }

  function setUserNameLabels(name, username) {
    const t = (name != null && name !== "" ? name : null) || username || "";
    const u = username || t;
    const desk = document.getElementById("user-name-desk");
    const menu = document.getElementById("user-name-menu");
    if (desk) {
      desk.textContent = t;
      desk.setAttribute("title", u);
    }
    if (menu) {
      menu.textContent = t;
      menu.setAttribute("title", u);
    }
  }

  function closeAccountDropdownDesk() {
    if (!accountDdDesk || !accountTriggerDesk) return;
    accountDdDesk.hidden = true;
    accountTriggerDesk.setAttribute("aria-expanded", "false");
  }

  function toggleAccountDropdownDesk() {
    if (!accountDdDesk || !accountTriggerDesk) return;
    const open = accountDdDesk.hidden;
    accountDdDesk.hidden = !open;
    accountTriggerDesk.setAttribute("aria-expanded", open ? "true" : "false");
  }

  if (accountTriggerDesk) {
    accountTriggerDesk.addEventListener("click", function (e) {
      e.stopPropagation();
      toggleAccountDropdownDesk();
    });
  }
  document.addEventListener("click", function (e) {
    if (accountMenuDesk && !accountMenuDesk.contains(e.target)) {
      closeAccountDropdownDesk();
    }
  });

  function showModalPair(show) {
    if (modalBackdrop) {
      modalBackdrop.hidden = !show;
    }
  }

  function closeAllModals() {
    if (modalProfile) {
      modalProfile.hidden = true;
    }
    if (modalSettings) {
      modalSettings.hidden = true;
    }
    showModalPair(false);
  }

  async function openProfileModal() {
    closeAccountDropdownDesk();
    if (MEDIA_MOBILE.matches && mobileMenu && mobileMenu.classList.contains("is-open")) {
      closeMobileMenu();
    }
    try {
      const r = await apiFetch("/api/user/profile", { method: "GET" });
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (r.status === 404) {
        window.location.href = "/login";
        return;
      }
      if (r.ok) {
        const j = await r.json();
        if (profileDisplayInput) {
          profileDisplayInput.value = j.display_name != null ? j.display_name : "";
        }
        if (profileEmailInput) {
          profileEmailInput.value = j.email != null && j.email !== "" ? j.email : "";
        }
        if (profileLoginLine) {
          profileLoginLine.textContent = "Utilizador de registo: " + (j.username || "");
        }
      }
    } catch (_) {}
    if (modalProfile) {
      modalProfile.hidden = false;
    }
    showModalPair(true);
  }

  async function openSettingsModal() {
    closeAccountDropdownDesk();
    if (MEDIA_MOBILE.matches && mobileMenu && mobileMenu.classList.contains("is-open")) {
      closeMobileMenu();
    }
    if (settingsBlockGlobal) {
      settingsBlockGlobal.hidden = true;
    }
    if (settingsBlockLlama) {
      settingsBlockLlama.hidden = true;
    }
    try {
      const r = await apiFetch("/api/user/settings", { method: "GET" });
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (r.status === 404) {
        window.location.href = "/login";
        return;
      }
      if (r.ok) {
        const j = await r.json();
        currentUserIsAdmin = isAdminFromApi(j.is_admin);
        setAdminNavVisible(currentUserIsAdmin);
        if (settingsBlockGlobal) {
          settingsBlockGlobal.hidden = !currentUserIsAdmin;
        }
        if (settingsBlockLlama) {
          settingsBlockLlama.hidden = !currentUserIsAdmin;
        }
        if (settingsModalHint) {
          if (currentUserIsAdmin) {
            settingsModalHint.textContent =
              "Como administrador: prompt global, prompt pessoal, e na secção abaixo os parâmetros de geração (tokens, temperatura, llama-server, etc.). Reinicia o Oráculo após mudar IP, porta ou «Usar llama-server».";
          } else {
            settingsModalHint.textContent =
              "Ajusta o teu system prompt; ele junta-se ao prompt global do serviço em cada geração. Os parâmetros do modelo são fixos na conta de utilizador.";
          }
        }
        if (settingsSystemPrompt) {
          settingsSystemPrompt.value = j.system_prompt != null ? j.system_prompt : "";
        }
        if (settingsGlobalSystemPrompt) {
          settingsGlobalSystemPrompt.value = j.global_system_prompt != null ? j.global_system_prompt : "";
        }
        if (currentUserIsAdmin && j.llama_server) {
          const L = j.llama_server;
          if (settingsLlamaEnabled) {
            settingsLlamaEnabled.checked = !!L.upstream_enabled;
          }
          if (settingsLlamaHost) {
            settingsLlamaHost.value = L.api_host != null ? String(L.api_host) : "";
          }
          if (settingsLlamaPort) {
            settingsLlamaPort.value = String(L.api_port != null ? L.api_port : 8080);
          }
          if (settingsLlamaNCtx) {
            settingsLlamaNCtx.value = String(L.n_ctx != null ? L.n_ctx : 4096);
          }
          if (settingsLlamaMaxTokens) {
            settingsLlamaMaxTokens.value = String(L.max_new_tokens != null ? L.max_new_tokens : 2048);
          }
          if (settingsLlamaTemp) {
            settingsLlamaTemp.value = String(L.temperature != null ? L.temperature : 0.8);
          }
          if (settingsLlamaTopP) {
            settingsLlamaTopP.value = String(L.top_p != null ? L.top_p : 0.9);
          }
          if (settingsLlamaRepeatPenalty) {
            settingsLlamaRepeatPenalty.value = String(L.repeat_penalty != null ? L.repeat_penalty : 1.15);
          }
          if (settingsLlamaRepeatLastN) {
            settingsLlamaRepeatLastN.value = String(L.repeat_last_n != null ? L.repeat_last_n : 512);
          }
          if (settingsLlamaReasoning) {
            settingsLlamaReasoning.value = L.reasoning != null ? String(L.reasoning) : "off";
          }
          if (settingsLlamaReasoningBudgetOn) {
            const rb = L.reasoning_budget != null ? Number(L.reasoning_budget) : 0;
            settingsLlamaReasoningBudgetOn.checked = rb !== 0;
          }
        }
      }
    } catch (_) {}
    if (modalSettings) {
      modalSettings.hidden = false;
    }
    showModalPair(true);
  }

  document.getElementById("open-profile-desk")?.addEventListener("click", function () {
    void openProfileModal();
  });
  document.getElementById("open-profile-m")?.addEventListener("click", function () {
    void openProfileModal();
  });
  document.getElementById("open-settings-desk")?.addEventListener("click", function () {
    void openSettingsModal();
  });
  document.getElementById("open-settings-m")?.addEventListener("click", function () {
    void openSettingsModal();
  });
  document.getElementById("modal-profile-close")?.addEventListener("click", closeAllModals);
  document.getElementById("profile-cancel")?.addEventListener("click", closeAllModals);
  document.getElementById("modal-settings-close")?.addEventListener("click", closeAllModals);
  document.getElementById("settings-cancel")?.addEventListener("click", closeAllModals);
  if (modalBackdrop) {
    modalBackdrop.addEventListener("click", closeAllModals);
  }

  document.getElementById("profile-save")?.addEventListener("click", async function () {
    const v = profileDisplayInput ? profileDisplayInput.value : "";
    const em = profileEmailInput ? profileEmailInput.value : "";
    try {
      const r = await apiFetch("/api/user/profile", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: v, email: em }),
      });
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (r.ok) {
        const j = await r.json();
        setUserNameLabels(j.name, j.username);
        closeAllModals();
        await loadSessionUser();
      }
    } catch (_) {}
  });

  document.getElementById("settings-save")?.addEventListener("click", async function () {
    const body = {
      system_prompt: settingsSystemPrompt ? settingsSystemPrompt.value : "",
    };
    if (currentUserIsAdmin) {
      body.global_system_prompt = settingsGlobalSystemPrompt
        ? settingsGlobalSystemPrompt.value
        : "";
      const lp = settingsLlamaPort ? parseInt(String(settingsLlamaPort.value), 10) : 8080;
      const lnCtx = settingsLlamaNCtx ? parseInt(String(settingsLlamaNCtx.value), 10) : 4096;
      const lMax = settingsLlamaMaxTokens ? parseInt(String(settingsLlamaMaxTokens.value), 10) : 2048;
      const lRn = settingsLlamaRepeatLastN ? parseInt(String(settingsLlamaRepeatLastN.value), 10) : 512;
      const lRb =
        settingsLlamaReasoningBudgetOn && settingsLlamaReasoningBudgetOn.checked ? -1 : 0;
      body.llama = {
        upstream_enabled: !!(settingsLlamaEnabled && settingsLlamaEnabled.checked),
        api_host: settingsLlamaHost ? String(settingsLlamaHost.value).trim() : "127.0.0.1",
        api_port: isNaN(lp) ? 8080 : lp,
        n_ctx: isNaN(lnCtx) ? 4096 : lnCtx,
        max_new_tokens: isNaN(lMax) ? 2048 : lMax,
        temperature: settingsLlamaTemp ? parseFloat(String(settingsLlamaTemp.value)) : 0.8,
        top_p: settingsLlamaTopP ? parseFloat(String(settingsLlamaTopP.value)) : 0.9,
        repeat_penalty: settingsLlamaRepeatPenalty
          ? parseFloat(String(settingsLlamaRepeatPenalty.value))
          : 1.15,
        repeat_last_n: isNaN(lRn) ? 512 : lRn,
        reasoning: settingsLlamaReasoning ? String(settingsLlamaReasoning.value) : "off",
        reasoning_budget: lRb,
      };
      if (isNaN(body.llama.temperature)) {
        body.llama.temperature = 0.8;
      }
      if (isNaN(body.llama.top_p)) {
        body.llama.top_p = 0.9;
      }
      if (isNaN(body.llama.repeat_penalty)) {
        body.llama.repeat_penalty = 1.15;
      }
    }
    try {
      const r = await apiFetch("/api/user/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (r.ok) {
        closeAllModals();
      }
    } catch (_) {}
  });

  async function loadSessionUser() {
    try {
      const r = await apiFetch("/api/auth/me", { method: "GET" });
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      const j = await r.json();
      if (j && j.authenticated) {
        setUserNameLabels(j.name, j.username);
        currentUserIsAdmin = isAdminFromApi(j.is_admin);
        setAdminNavVisible(currentUserIsAdmin);
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

  /** Se o fundo está a menos disto que isto em px, considera-se seguir o stream até ao fundo — valor baixo para um deslize mínimo cortar follow. */
  const LOG_BOTTOM_THRESHOLD_PX = 28;
  let logUserFollowingBottom = true;
  /** Ignora handlers de scroll em efeitos de ``scrollTop`` impostos por JS para não repetir estado. */
  let logSuppressScrollSemantics = false;

  function getScrollContainer() {
    return mainScrollEl || logEl;
  }

  function isLogNearBottom() {
    const el = getScrollContainer();
    if (!el) return true;
    const gap = el.scrollHeight - el.clientHeight - el.scrollTop;
    return gap <= LOG_BOTTOM_THRESHOLD_PX;
  }

  function scrollLog() {
    const el = getScrollContainer();
    if (!el) return;
    logSuppressScrollSemantics = true;
    el.scrollTop = el.scrollHeight;
    queueMicrotask(function () {
      logSuppressScrollSemantics = false;
    });
  }

  function scrollLogIfFollowing() {
    if (logUserFollowingBottom) {
      scrollLog();
    }
  }

  (function attachLogScrollFollowSignals() {
    const el = getScrollContainer();
    if (!el) return;

    el.addEventListener(
      "scroll",
      function () {
        if (logSuppressScrollSemantics) return;
        logUserFollowingBottom = isLogNearBottom();
      },
      { passive: true }
    );

    /** Roda imediatamente para cima mesmo ligeira — pára seguir até voltar ao fim manualmente ou próximo envio. */
    el.addEventListener(
      "wheel",
      function (e) {
        var dy = e.deltaY;
        if (e.deltaMode === 1) dy *= 16;
        else if (e.deltaMode === 2) dy *= Math.max(el.clientHeight, 100);
        if (dy < -1) logUserFollowingBottom = false;
      },
      { passive: true }
    );

    /* Telemóveis: primeiro deslizar que mover scrollTop suficiente também corta o follow. */
    var touchAnchScrollTop = null;
    el.addEventListener(
      "touchstart",
      function () {
        touchAnchScrollTop = el.scrollTop;
      },
      { passive: true }
    );
    el.addEventListener(
      "touchmove",
      function () {
        if (touchAnchScrollTop == null) return;
        if (Math.abs(el.scrollTop - touchAnchScrollTop) > 3) logUserFollowingBottom = false;
      },
      { passive: true }
    );
    function clearTouchAnch() {
      touchAnchScrollTop = null;
    }
    el.addEventListener("touchend", clearTouchAnch, { passive: true });
    el.addEventListener("touchcancel", clearTouchAnch, { passive: true });
  })();

  const COPY_ICON_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';

  const CHECK_ICON_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>';

  const MSG_STAT_SVG_TOKENS =
    '<svg class="msg__stat-ico" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="5" rx="4" ry="2"/><path d="M8 5v4c0 1.1 1.8 2 4 2s4-.9 4-2V5"/><path d="M8 9v4c0 1.1 1.8 2 4 2s4-.9 4-2V9"/><path d="M8 13v3c0 1.1 1.8 2 4 2s4-.9 4-2v-3"/></svg>';

  const MSG_STAT_SVG_CLOCK =
    '<svg class="msg__stat-ico" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>';

  const MSG_STAT_SVG_TPS =
    '<svg class="msg__stat-ico" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20h16"/><path d="M6 20a8 6 0 0 1 12 0"/><path d="M12 10V6"/><path d="M12 10l3 2"/></svg>';

  function formatGenSec(s) {
    if (s == null || !Number.isFinite(Number(s)) || Number(s) < 0) {
      return "—";
    }
    const n = Number(s);
    return n.toFixed(1).replace(/\.0$/, "") + "s";
  }

  function createAssistantStatsEl() {
    const root = document.createElement("div");
    root.className = "msg__stats msg__stats--empty";
    root.setAttribute("aria-hidden", "true");
    function line(icon, key) {
      const span = document.createElement("span");
      span.className = "msg__stat";
      const holder = document.createElement("div");
      holder.innerHTML = icon;
      const val = document.createElement("span");
      val.className = "msg__stat-val";
      val.dataset.m = key;
      span.appendChild(holder.firstElementChild);
      span.appendChild(val);
      return span;
    }
    root.appendChild(line(MSG_STAT_SVG_TOKENS, "tok"));
    root.appendChild(line(MSG_STAT_SVG_CLOCK, "sec"));
    root.appendChild(line(MSG_STAT_SVG_TPS, "tps"));
    return root;
  }

  function setAssistantGenStats(root, st) {
    if (!root) return;
    if (!st) return;
    const has =
      (st.output_tokens != null && st.output_tokens !== undefined) ||
      (st.gen_seconds != null && st.gen_seconds !== undefined) ||
      (st.tokens_per_sec != null && st.tokens_per_sec !== undefined);
    if (!has) return;
    root.classList.remove("msg__stats--empty");
    root.setAttribute("aria-hidden", "false");
    const tok = root.querySelector('.msg__stat-val[data-m="tok"]');
    const sec = root.querySelector('.msg__stat-val[data-m="sec"]');
    const tps = root.querySelector('.msg__stat-val[data-m="tps"]');
    if (st.output_tokens != null && st.output_tokens !== undefined) {
      if (tok) tok.textContent = String(st.output_tokens) + " tokens";
    } else if (tok) {
      tok.textContent = "—";
    }
    if (st.gen_seconds != null && st.gen_seconds !== undefined) {
      if (sec) sec.textContent = formatGenSec(Number(st.gen_seconds));
    } else if (sec) {
      sec.textContent = "—";
    }
    if (st.tokens_per_sec != null && st.tokens_per_sec !== undefined) {
      if (tps) tps.textContent = Number(st.tokens_per_sec).toFixed(2) + " t/s";
    } else if (tps) {
      tps.textContent = "—";
    }
  }

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

  function appendMessage(role, content, genStats) {
    const stack = document.createElement("div");
    stack.className =
      "msg-stack " + (role === "assistant" ? "msg-stack--assistant" : "msg-stack--user");
    const bubble = document.createElement("div");
    bubble.className = "msg " + role;
    const textEl = document.createElement("div");
    textEl.className = "msg__text";
    renderMsgMarkdown(textEl, content);
    const copyBtn = document.createElement("button");
    bindCopyButton(copyBtn, () => getMsgCopyText(textEl), stack);
    bubble.appendChild(textEl);
    stack.appendChild(bubble);
    if (role === "assistant") {
      const bar = document.createElement("div");
      bar.className = "msg-stack__bar";
      bar.appendChild(copyBtn);
      bar.appendChild(createAssistantStatsEl());
      stack.appendChild(bar);
      if (genStats) {
        const statsEl = bar.querySelector(".msg__stats");
        if (statsEl) {
          setAssistantGenStats(statsEl, genStats);
        }
      }
    } else {
      stack.appendChild(copyBtn);
    }
    bindMessageTapToRevealCopy(stack, bubble);
    logInner.appendChild(stack);
    updateEmptyState();
    scrollLog();
  }

  function appendAssistantStreaming() {
    const stack = document.createElement("div");
    stack.className = "msg-stack msg-stack--assistant";
    const bubble = document.createElement("div");
    bubble.className = "msg assistant streaming";
    const textEl = document.createElement("div");
    textEl.className = "msg__text";
    const copyBtn = document.createElement("button");
    bindCopyButton(copyBtn, () => getMsgCopyText(textEl), stack);
    const statsEl = createAssistantStatsEl();
    const bar = document.createElement("div");
    bar.className = "msg-stack__bar";
    bar.appendChild(copyBtn);
    bar.appendChild(statsEl);
    bubble.appendChild(textEl);
    stack.appendChild(bubble);
    stack.appendChild(bar);
    bindMessageTapToRevealCopy(stack, bubble);
    logInner.appendChild(stack);
    updateEmptyState();
    scrollLogIfFollowing();
    return { stack: stack, div: bubble, textEl: textEl, statsEl: statsEl };
  }

  function clearChat() {
    setCopyPinnedStack(null);
    history.length = 0;
    logInner.innerHTML = "";
    inputEl.value = "";
    collapseComposerToSingleRow();
    delete sendBtn.dataset.busy;
    updateSendState();
    updateEmptyState();
    if (mobileMenu && mobileMenu.classList.contains("is-open")) closeMobileMenu();
    inputEl.focus();
  }

  function closeAllSessionMenus() {
    document.querySelectorAll(".session-list__menu.is-open").forEach(function (m) {
      m.classList.remove("is-open");
      const kb = m.querySelector(".session-list__kebab");
      if (kb) {
        kb.setAttribute("aria-expanded", "false");
      }
    });
  }

  function buildSessionItem(s) {
    const li = document.createElement("li");
    li.className =
      "session-list__item" +
      (String(s.id) === String(currentSessionId) ? " session-list__item--active" : "");
    li.dataset.sid = String(s.id);
    const row = document.createElement("div");
    row.className = "session-list__row";
    const isEditing = editingSessionId != null && String(editingSessionId) === String(s.id);

    if (isEditing) {
      const entry = document.createElement("div");
      entry.className = "session-list__entry";
      const inp = document.createElement("input");
      inp.type = "text";
      inp.className = "session-list__title-input";
      inp.value = s.title || "Conversa";
      inp.setAttribute("aria-label", "Nome da conversa");
      inp.setAttribute("autocomplete", "off");
      entry.appendChild(inp);
      row.appendChild(entry);
    } else {
      const entry = document.createElement("div");
      entry.className = "session-list__entry";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "session-list__btn";
      const fullLabel = (s.title || "Conversa").trim() || "Conversa";
      btn.setAttribute("title", fullLabel);
      btn.setAttribute("aria-label", fullLabel);
      btn.textContent = fullLabel;
      const menu = document.createElement("div");
      menu.className = "session-list__menu";
      const kebab = document.createElement("button");
      kebab.type = "button";
      kebab.className = "session-list__kebab";
      kebab.setAttribute("aria-label", "Mais opções");
      kebab.setAttribute("aria-haspopup", "true");
      kebab.setAttribute("aria-expanded", "false");
      kebab.appendChild(document.createTextNode("⋯"));
      const dd = document.createElement("div");
      dd.className = "session-list__dropdown";
      dd.setAttribute("role", "menu");
      const renameBtn = document.createElement("button");
      renameBtn.type = "button";
      renameBtn.className = "session-list__dd-item";
      renameBtn.setAttribute("role", "menuitem");
      renameBtn.dataset.action = "rename";
      renameBtn.innerHTML =
        '<svg class="session-list__dd-ico" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg><span>Renomear</span>';
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "session-list__dd-item";
      removeBtn.setAttribute("role", "menuitem");
      removeBtn.setAttribute("aria-label", "Remover");
      removeBtn.dataset.action = "remove";
      removeBtn.innerHTML =
        '<svg class="session-list__dd-ico" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg><span>Remover</span>';
      dd.appendChild(renameBtn);
      dd.appendChild(removeBtn);
      menu.appendChild(kebab);
      menu.appendChild(dd);
      entry.appendChild(btn);
      entry.appendChild(menu);
      row.appendChild(entry);
    }

    li.appendChild(row);
    return li;
  }

  function startSessionRename(sid) {
    closeAllSessionMenus();
    editingSessionId = String(sid);
    renderSessionList();
    requestAnimationFrame(function () {
      const root = MEDIA_MOBILE.matches ? sessionListMenuEl : sessionListEl;
      const inp =
        (root && root.querySelector(".session-list__title-input")) ||
        document.querySelector(".session-list__title-input");
      if (inp) {
        inp.focus();
        inp.select();
      }
    });
  }

  async function commitSessionRename(sid, rawValue) {
    if (savingSessionRename) return;
    const sidStr = String(sid);
    if (editingSessionId == null || String(editingSessionId) !== sidStr) {
      return;
    }
    const trimmed = (rawValue || "").trim();
    const prev = (lastSessions.find(function (x) { return String(x.id) === sidStr; }) || {})
      .title;
    if (trimmed === (prev || "") || !trimmed) {
      editingSessionId = null;
      renderSessionList();
      return;
    }
    savingSessionRename = true;
    editingSessionId = null;
    renderSessionList();
    try {
      const r = await apiFetch("/api/sessions/" + encodeURIComponent(sid), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: trimmed }),
      });
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (r.ok) {
        const j = await r.json();
        for (let i = 0; i < lastSessions.length; i++) {
          if (String(lastSessions[i].id) === sidStr) {
            lastSessions[i].title = j.title;
            break;
          }
        }
        renderSessionList();
        return;
      }
      editingSessionId = sidStr;
      renderSessionList();
      requestAnimationFrame(function () {
        const root = MEDIA_MOBILE.matches ? sessionListMenuEl : sessionListEl;
        const inp =
          (root && root.querySelector('.session-list__item[data-sid="' + sidStr + '"] .session-list__title-input')) ||
          document.querySelector('.session-list__item[data-sid="' + sidStr + '"] .session-list__title-input');
        if (inp) {
          inp.value = trimmed;
          inp.focus();
          inp.select();
        }
      });
    } catch (_) {
      editingSessionId = sidStr;
      renderSessionList();
      requestAnimationFrame(function () {
        const root = MEDIA_MOBILE.matches ? sessionListMenuEl : sessionListEl;
        const inp =
          (root && root.querySelector('.session-list__item[data-sid="' + sidStr + '"] .session-list__title-input')) ||
          document.querySelector('.session-list__item[data-sid="' + sidStr + '"] .session-list__title-input');
        if (inp) {
          inp.value = trimmed;
          inp.focus();
          inp.select();
        }
      });
    } finally {
      savingSessionRename = false;
    }
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
      closeMobileMenuIfOpen();
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
        const genStats =
          m.role === "assistant"
            ? {
                output_tokens: m.output_tokens,
                gen_seconds: m.gen_seconds,
                tokens_per_sec: m.tokens_per_sec,
              }
            : null;
        appendMessage(m.role, m.content, genStats);
      }
    }
    updateEmptyState();
    scrollLog();
    closeMobileMenuIfOpen();
    void loadSessionList();
  }

  function onSessionListClick(e) {
    if (e.target.closest(".session-list__title-input")) {
      e.stopPropagation();
      return;
    }
    if (e.target.closest(".session-list__kebab")) {
      e.stopPropagation();
      e.preventDefault();
      const menu = e.target.closest(".session-list__menu");
      if (menu) {
        const wasOpen = menu.classList.contains("is-open");
        closeAllSessionMenus();
        if (!wasOpen) {
          menu.classList.add("is-open");
          const kb = menu.querySelector(".session-list__kebab");
          if (kb) {
            kb.setAttribute("aria-expanded", "true");
          }
        }
      }
      return;
    }
    const act = e.target.closest("[data-action]");
    if (act) {
      e.stopPropagation();
      e.preventDefault();
      if (act.dataset.action === "remove") {
        closeAllSessionMenus();
        return;
      }
      if (act.dataset.action === "rename") {
        const li = act.closest("li");
        if (li && li.dataset.sid != null) {
          startSessionRename(li.dataset.sid);
        }
      }
      return;
    }
    if (e.target.closest(".session-list__menu")) {
      e.stopPropagation();
      return;
    }
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

  function onGlobalCloseSessionMenu(e) {
    if (e && e.target && e.target.closest && e.target.closest(".session-list__menu")) {
      return;
    }
    closeAllSessionMenus();
  }
  document.addEventListener("click", onGlobalCloseSessionMenu);

  function onSessionListFocusOut(e) {
    const t = e.target;
    if (!t || !t.classList || !t.classList.contains("session-list__title-input")) {
      return;
    }
    const li = t.closest("li");
    if (!li || li.dataset.sid == null) return;
    void commitSessionRename(li.dataset.sid, t.value);
  }
  if (sessionListEl) {
    sessionListEl.addEventListener("focusout", onSessionListFocusOut);
  }
  if (sessionListMenuEl) {
    sessionListMenuEl.addEventListener("focusout", onSessionListFocusOut);
  }

  function onSessionListKeydown(e) {
    if (!e.target.classList.contains("session-list__title-input")) {
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      e.target.blur();
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      editingSessionId = null;
      renderSessionList();
    }
  }
  if (sessionListEl) {
    sessionListEl.addEventListener("keydown", onSessionListKeydown, true);
  }
  if (sessionListMenuEl) {
    sessionListMenuEl.addEventListener("keydown", onSessionListKeydown, true);
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
    collapseComposerToSingleRow();
    updateSendState();

    appendMessage("user", text);
    history.push({ role: "user", content: text });

    sendBtn.dataset.busy = "1";
    updateSendState();

    const { div: assistantDiv, textEl, statsEl } = appendAssistantStreaming();
    let lastText = "";
    let endReason = "none";
    let lastJobState = null;

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
        renderMsgMarkdown(textEl, lastText);
        scrollLogIfFollowing();
        if (st.status === "done") {
          endReason = "done";
          lastJobState = st;
          break;
        }
        if (st.status === "error") {
          throw new Error(st.error || "Erro no modelo");
        }
        if (st.status === "cancelled") {
          endReason = "cancelled";
          lastJobState = st;
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
        textEl.classList.remove("msg__text--md");
        textEl.removeAttribute("data-raw-md");
        textEl.textContent = "Erro: " + e.message;
      }
    } finally {
      if (endReason === "done") {
        history.push({ role: "assistant", content: lastText });
      } else if (endReason === "cancelled" || endReason === "abort") {
        if (lastText.trim().length) {
          renderMsgMarkdown(textEl, lastText);
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
      if (statsEl && lastJobState && logInner && logInner.contains(statsEl)) {
        setAssistantGenStats(statsEl, lastJobState);
      }
      currentJobId = null;
      releaseScreenWakeLock();
      if (streamAborter === ac) {
        streamAborter = null;
      }
      delete sendBtn.dataset.busy;
      updateSendState();
      collapseComposerToSingleRow();
      void loadSessionList();
      void tickGenerationStatus();
      /* Telemóvel (web): foco abre teclado e encolhe o ecrã; desktop mantém foco para continuar a escrever. */
      if (MEDIA_MOBILE.matches) {
        inputEl.blur();
      } else if (document.visibilityState === "visible") {
        try {
          inputEl.focus({ preventScroll: true });
        } catch (_e) {
          inputEl.focus();
        }
      }
    }
  }

  sendBtn.addEventListener("click", function onSendOrStop() {
    if (sendBtn.dataset.busy === "1") {
      abortStream();
      return;
    }
    send();
  });
  composerPlusBtn?.addEventListener("click", () => {
    inputEl?.focus();
  });

  window.addEventListener(
    "resize",
    function onComposerResize() {
      autoResizeInput();
    },
    { passive: true }
  );

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (sendBtn.dataset.busy === "1") return;
      if (!sendBtn.disabled) send();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (modalBackdrop && !modalBackdrop.hidden) {
      closeAllModals();
      return;
    }
    if (mobileMenu && mobileMenu.classList.contains("is-open")) {
      closeMobileMenu();
    }
  });

  MEDIA_MOBILE.addEventListener("change", (e) => {
    if (!e.matches && mobileMenu && mobileMenu.classList.contains("is-open")) {
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
    startGenerationStatusPolling();
  })();
})();
