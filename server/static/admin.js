/**
 * Oráculo Kiaiá — painel de admin (utilizadores, sessões, leitura de histórico)
 */
(function () {
  "use strict";

  const listEl = document.getElementById("admin-user-list");
  const errEl = document.getElementById("admin-user-err");
  const sessionsTitle = document.getElementById("admin-sessions-title");
  const sessionsPlaceholder = document.getElementById("admin-sessions-placeholder");
  const sessionListEl = document.getElementById("admin-session-list");
  const viewModal = document.getElementById("admin-view-modal");
  const viewBackdrop = document.getElementById("admin-view-backdrop");
  const viewLog = document.getElementById("admin-view-log");
  const viewTitle = document.getElementById("admin-view-title");
  const viewClose = document.getElementById("admin-view-close");

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

  function apiFetch(path, init) {
    const o = init ? Object.assign({}, init) : {};
    o.credentials = o.credentials || "same-origin";
    return fetch(apiUrl(path), o);
  }

  if (typeof marked !== "undefined" && typeof marked.setOptions === "function") {
    marked.setOptions({ gfm: true, breaks: true, mangle: false, headerIds: false });
  }

  let dompurifyHooked = false;
  function renderMdInto(el, raw) {
    const md = raw == null ? "" : String(raw);
    if (!md) {
      el.textContent = "";
      return;
    }
    if (typeof marked === "undefined" || typeof marked.parse !== "function") {
      el.textContent = md;
      return;
    }
    if (typeof DOMPurify === "undefined" || typeof DOMPurify.sanitize !== "function") {
      el.textContent = md;
      return;
    }
    if (!dompurifyHooked && typeof DOMPurify.addHook === "function") {
      dompurifyHooked = true;
      DOMPurify.addHook("afterSanitizeAttributes", function (node) {
        if (node.tagName === "A" && node.hasAttribute("href")) {
          node.setAttribute("target", "_blank");
          node.setAttribute("rel", "noopener noreferrer");
        }
      });
    }
    try {
      el.classList.add("msg__text--md", "msg__text");
      const html = marked.parse(md);
      el.innerHTML = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    } catch (_e) {
      el.textContent = md;
    }
  }

  let selectedUserId = null;
  let selectedLabel = "";
  let pollTimer = null;

  function showErr(msg) {
    if (!errEl) return;
    if (msg) {
      errEl.textContent = msg;
      errEl.hidden = false;
    } else {
      errEl.textContent = "";
      errEl.hidden = true;
    }
  }

  function renderUserRow(u) {
    const li = document.createElement("li");
    li.className = "admin-user-row" + (selectedUserId === u.id ? " is-selected" : "");
    li.dataset.uid = String(u.id);
    li.setAttribute("role", "button");
    li.setAttribute("tabindex", "0");
    li.setAttribute(
      "aria-label",
      (u.display_name || u.username) + (u.using_server ? " — a gerar" : u.online ? " — ligado" : "")
    );

    const dots = document.createElement("span");
    dots.className = "admin-user-dots";
    if (u.online) {
      const dOn = document.createElement("span");
      dOn.className = "admin-dot admin-dot--on";
      dOn.title = "Ligado (atividade recente)";
      dOn.setAttribute("aria-hidden", "true");
      dots.appendChild(dOn);
    }
    if (u.using_server) {
      const dB = document.createElement("span");
      dB.className = "admin-dot admin-dot--busy";
      dB.title = "A utilizar o modelo agora";
      dB.setAttribute("aria-hidden", "true");
      dots.appendChild(dB);
    }

    const main = document.createElement("div");
    main.className = "admin-user-row__main";
    const name = document.createElement("span");
    name.className = "admin-user-row__name";
    name.textContent = u.display_name || u.username;
    const sub = document.createElement("span");
    sub.className = "admin-user-row__sub";
    sub.textContent = u.username + (u.is_admin ? " · admin" : "");
    main.appendChild(name);
    main.appendChild(sub);

    li.appendChild(dots);
    li.appendChild(main);
    return li;
  }

  async function loadUsers() {
    try {
      const r = await apiFetch("/api/admin/users");
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (r.status === 403) {
        window.location.href = "/";
        return;
      }
      if (!r.ok) {
        showErr("Não foi possível carregar utilizadores.");
        return;
      }
      showErr("");
      const j = await r.json();
      const users = j.users || [];
      if (listEl) {
        listEl.innerHTML = "";
        users.forEach(function (u) {
          listEl.appendChild(renderUserRow(u));
        });
        updateUserRowHighlight();
      }
    } catch (e) {
      showErr("Sem ligação ao servidor.");
    }
  }

  function updateUserRowHighlight() {
    if (!listEl) return;
    listEl.querySelectorAll(".admin-user-row").forEach(function (row) {
      const n = parseInt(row.dataset.uid, 10);
      row.classList.toggle(
        "is-selected",
        selectedUserId != null && !isNaN(n) && n === selectedUserId
      );
    });
  }

  function selectUser(uid, label) {
    selectedUserId = uid;
    selectedLabel = label || String(uid);
    if (sessionsTitle) {
      sessionsTitle.textContent = "Sessões — " + selectedLabel;
    }
    updateUserRowHighlight();
    void loadSessions(uid);
  }

  async function loadSessions(uid) {
    if (sessionsPlaceholder) {
      sessionsPlaceholder.hidden = true;
    }
    if (sessionListEl) {
      sessionListEl.innerHTML = "";
      sessionListEl.hidden = false;
    }
    try {
      const r = await apiFetch("/api/admin/users/" + encodeURIComponent(String(uid)) + "/sessions");
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (!r.ok) {
        if (sessionListEl) {
          sessionListEl.innerHTML = "<li class='admin-err-txt'>Erro ao listar sessões.</li>";
        }
        return;
      }
      const j = await r.json();
      const sessions = j.sessions || [];
      if (!sessionListEl) {
        return;
      }
      if (sessions.length === 0) {
        sessionListEl.innerHTML = "<li class='admin-err-txt'>Nenhuma sessão.</li>";
        return;
      }
      sessions.forEach(function (s) {
        const li = document.createElement("li");
        li.className = "admin-session-row";
        const sid = s.id;
        const title = (s.title || "Conversa").trim() || "Conversa";
        const titleEl = document.createElement("div");
        titleEl.className = "admin-session-row__title";
        titleEl.textContent = title;
        li.appendChild(titleEl);
        const tTok = s.total_output_tokens != null ? Number(s.total_output_tokens) : 0;
        const tSec = s.total_gen_seconds != null ? Number(s.total_gen_seconds) : 0;
        if (tTok > 0 || tSec > 0) {
          const tps = tSec > 0 && tTok > 0 ? (tTok / tSec).toFixed(2) : "—";
          const meta = document.createElement("div");
          meta.className = "admin-session-row__meta";
          const secStr =
            tSec > 0 && Number.isFinite(tSec)
              ? tSec.toFixed(1).replace(/\.0$/, "") + "s"
              : "—";
          meta.textContent = tTok + " tokens · " + secStr + " · " + tps + " t/s";
          li.appendChild(meta);
        }
        li.title = "Abrir histórico";
        li.setAttribute("role", "button");
        li.setAttribute("tabindex", "0");
        li.addEventListener("click", function () {
          void openSession(uid, sid, title);
        });
        li.addEventListener("keydown", function (e) {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            void openSession(uid, sid, title);
          }
        });
        sessionListEl.appendChild(li);
      });
    } catch (e) {
      if (sessionListEl) {
        sessionListEl.innerHTML = "<li class='admin-err-txt'>Erro de rede.</li>";
      }
    }
  }

  async function openSession(uid, sessionId, title) {
    if (!viewModal || !viewLog || !viewTitle) return;
    try {
      const r = await apiFetch(
        "/api/admin/users/" + encodeURIComponent(String(uid)) + "/sessions/" + encodeURIComponent(String(sessionId))
      );
      if (r.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (!r.ok) {
        return;
      }
      const d = await r.json();
      viewTitle.textContent = (d.title || title || "Sessão").trim();
      viewLog.innerHTML = "";
      const msgs = d.messages || [];
      msgs.forEach(function (m) {
        const role = m.role;
        if (role !== "user" && role !== "assistant" && role !== "system") {
          return;
        }
        const row = document.createElement("div");
        row.className = "admin-view-msg admin-view-msg--" + (role === "user" ? "user" : role === "assistant" ? "assistant" : "sys");
        const label = document.createElement("div");
        label.className = "admin-view-msg__role";
        label.textContent = role === "user" ? "Utilizador" : role === "assistant" ? "Assistente" : "Sistema";
        const body = document.createElement("div");
        body.className = "admin-view-msg__body";
        if (role === "assistant") {
          renderMdInto(body, m.content);
        } else {
          body.classList.add("msg__text");
          body.textContent = m.content != null ? String(m.content) : "";
        }
        row.appendChild(label);
        row.appendChild(body);
        if (
          role === "assistant" &&
          (m.output_tokens != null || m.gen_seconds != null || m.tokens_per_sec != null)
        ) {
          const stEl = document.createElement("div");
          stEl.className = "admin-view-msg__stats";
          const parts = [];
          if (m.output_tokens != null) {
            parts.push(String(m.output_tokens) + " tokens");
          }
          if (m.gen_seconds != null) {
            const n = Number(m.gen_seconds);
            if (Number.isFinite(n) && n >= 0) {
              parts.push(n.toFixed(1).replace(/\.0$/, "") + "s");
            }
          }
          if (m.tokens_per_sec != null) {
            parts.push(Number(m.tokens_per_sec).toFixed(2) + " t/s");
          }
          stEl.textContent = parts.join(" · ");
          row.appendChild(stEl);
        }
        viewLog.appendChild(row);
      });
      viewModal.hidden = false;
      if (viewBackdrop) {
        viewBackdrop.hidden = false;
        viewBackdrop.setAttribute("aria-hidden", "false");
      }
      viewModal.setAttribute("aria-hidden", "false");
    } catch (_) {}
  }

  function closeView() {
    if (viewModal) {
      viewModal.hidden = true;
      viewModal.setAttribute("aria-hidden", "true");
    }
    if (viewBackdrop) {
      viewBackdrop.hidden = true;
      viewBackdrop.setAttribute("aria-hidden", "true");
    }
  }

  if (listEl) {
    listEl.addEventListener("click", function (e) {
      const li = e.target.closest(".admin-user-row");
      if (!li) return;
      const uid = parseInt(li.dataset.uid, 10);
      if (isNaN(uid)) return;
      const name = li.querySelector(".admin-user-row__name");
      selectUser(uid, name ? name.textContent : null);
    });
  }

  if (viewClose) {
    viewClose.addEventListener("click", closeView);
  }
  if (viewBackdrop) {
    viewBackdrop.addEventListener("click", closeView);
  }
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && viewModal && !viewModal.hidden) {
      closeView();
    }
  });

  if (viewModal) {
    viewModal.addEventListener("click", function (e) {
      if (e.target === viewModal) {
        closeView();
      }
    });
  }

  void loadUsers();
  pollTimer = setInterval(loadUsers, 3000);
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible") {
      void loadUsers();
    }
  });
})();
