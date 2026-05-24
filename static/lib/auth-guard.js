/**
 * HomeHub auth guard — protects camera-related pages.
 *
 * Include in <head> BEFORE other JS:
 *   <script src="/lib/auth-guard.js"></script>
 *
 * Behavior:
 *   1. Calls /api/auth/whoami
 *   2. If auth_enabled && !user → redirects to /login.html?next=<here>
 *   3. If logged in → injects user info into the page top-bar:
 *        a) If a `#user-slot` element exists, fills it (preferred — page
 *           controls placement and styling).
 *        b) Else, tries to append into a known top-bar container
 *           (.hdr-right, header > nav, header).
 *        c) Else falls back to a fixed top-right widget (last resort).
 *
 * Pages add `<div id="user-slot"></div>` inside their top-bar to opt into (a).
 */
(function () {
  // Not on the login page itself
  if (location.pathname === "/login.html") return;

  // Hide the body until we know the auth state — avoids "flash of protected content"
  document.documentElement.style.visibility = "hidden";

  fetch("/api/auth/whoami", { cache: "no-store" })
    .then((r) => r.json())
    .then((d) => {
      if (d.auth_enabled && !d.user) {
        const next = location.pathname + location.search;
        location.replace("/login.html?next=" + encodeURIComponent(next));
        return;
      }
      document.documentElement.style.visibility = "";
      window.HOMEHUB_AUTH = d;
      injectUserWidget(d);
    })
    .catch(() => {
      document.documentElement.style.visibility = "";
    });

  function injectUserWidget(authData) {
    if (!authData.auth_enabled || !authData.user) return;
    if (location.pathname === "/account.html") return;

    // Inject CSS once.
    if (!document.getElementById("auth-guard-css")) {
      const style = document.createElement("style");
      style.id = "auth-guard-css";
      style.textContent = `
        .auth-user-info {
          display: inline-flex; align-items: center; gap: 6px;
          font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          color: var(--muted, #8b949e);
          padding: 4px 8px;
          border-radius: 6px;
          white-space: nowrap;
        }
        .auth-user-info .auth-user-name {
          color: var(--text, #e6edf3);
          font-weight: 500;
        }
        .auth-user-info a {
          color: var(--accent, #58a6ff);
          text-decoration: none;
        }
        .auth-user-info a:hover { text-decoration: underline; }
        .auth-user-info .auth-sep {
          opacity: .35;
        }
        @media (max-width: 600px) {
          .auth-user-info { font-size: 11px; padding: 2px 4px; gap: 4px; }
          .auth-user-info .auth-sep { display: none; }
        }
        /* Fallback floating widget when no slot is available */
        #auth-user-floating {
          position: fixed; top: 8px; right: 12px; z-index: 9999;
          background: rgba(22,27,34,.85);
          border: 1px solid var(--border, #30363d);
          backdrop-filter: blur(4px);
        }
      `;
      document.head.appendChild(style);
    }

    const html =
      `<span>👤</span>` +
      `<span class="auth-user-name">${escapeHtml(authData.user)}</span>` +
      `<a href="/account.html">Account</a>` +
      `<a href="#" class="auth-logout">Logout</a>`;

    // Try, in order: explicit slot, .hdr-right, header > nav/.actions/.nav, header.
    const target =
      document.getElementById("user-slot") ||
      document.querySelector("header .hdr-right") ||
      document.querySelector("header .actions") ||
      document.querySelector("header nav") ||
      document.querySelector("header .nav") ||
      document.querySelector(".header > div:last-child");

    let widget;
    if (target) {
      widget = document.createElement("div");
      widget.className = "auth-user-info";
      widget.innerHTML = html;
      target.appendChild(widget);
    } else {
      // Last-resort fallback: floating widget (only if no header nav was found).
      widget = document.createElement("div");
      widget.id = "auth-user-floating";
      widget.className = "auth-user-info";
      widget.innerHTML = html;
      document.body.appendChild(widget);
    }

    widget.querySelector(".auth-logout").addEventListener("click", async (e) => {
      e.preventDefault();
      try {
        await fetch("/api/auth/logout", { method: "POST" });
      } catch {}
      location.href = "/login.html";
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[c]);
  }
})();
