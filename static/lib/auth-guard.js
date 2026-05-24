/**
 * HomeHub auth guard — синхронен redirect ако auth е enabled и не сме logged-in.
 *
 * Включи на всеки protected HTML page ВЪВ <head>, ПРЕДИ останалия JS:
 *   <script src="/lib/auth-guard.js"></script>
 *
 * Server-side enforcement остава за /api/* (middleware). Този скрипт:
 *   - извиква /api/auth/whoami
 *   - ако auth_enabled=true && !user  →  redirect към /login.html?next=<тук>
 *   - inline (не async): блокира rendering-а на UI до приключване
 *
 * За API endpoints — ако backend върне 401, никой UI не може да caches-ва нищо.
 */
(function () {
  // Не пускай guard-а на самата login страница (тя си има логика)
  if (location.pathname === "/login.html") return;

  // Synchronous fetch е невъзможен; ползваме async fetch и крием body-то
  // докато не получим резултат — за да избегнем "flash of protected content".
  document.documentElement.style.visibility = "hidden";

  fetch("/api/auth/whoami", { cache: "no-store" })
    .then((r) => r.json())
    .then((d) => {
      if (d.auth_enabled && !d.user) {
        const next = location.pathname + location.search;
        location.replace("/login.html?next=" + encodeURIComponent(next));
        return; // Не показвай UI
      }
      document.documentElement.style.visibility = "";
      window.HOMEHUB_AUTH = d;   // {auth_enabled, user}
      injectUserWidget(d);
    })
    .catch(() => {
      // При мрежова грешка показваме UI (auth може да е disabled)
      document.documentElement.style.visibility = "";
    });

  /**
   * Inject малък user widget top-right (само ако auth е enabled и сме logged in
   * и страницата НЕ е /account.html — там вече има account info).
   * Минимален CSS, fixed position — не пречи на съществуващия layout.
   */
  function injectUserWidget(authData) {
    if (!authData.auth_enabled || !authData.user) return;
    if (location.pathname === "/account.html") return;
    if (document.getElementById("homehub-user-widget")) return;

    const css = `
      #homehub-user-widget {
        position: fixed; top: 8px; right: 12px; z-index: 9999;
        font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: rgba(22,27,34,.85); color: #e6edf3;
        border: 1px solid #30363d; border-radius: 6px;
        padding: 4px 10px; backdrop-filter: blur(4px);
        display: flex; align-items: center; gap: 8px;
      }
      #homehub-user-widget a {
        color: #58a6ff; text-decoration: none;
      }
      #homehub-user-widget a:hover { text-decoration: underline; }
      #homehub-user-widget .sep { color: #30363d; }
      @media (max-width: 480px) {
        #homehub-user-widget { font-size: 11px; padding: 3px 8px; gap: 6px; }
      }
    `;
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);

    const widget = document.createElement("div");
    widget.id = "homehub-user-widget";
    widget.innerHTML =
      `<span>👤 ${escapeHtml(authData.user)}</span>` +
      `<span class="sep">·</span>` +
      `<a href="/account.html">Account</a>` +
      `<span class="sep">·</span>` +
      `<a href="#" id="homehub-logout">Logout</a>`;
    document.body.appendChild(widget);

    document.getElementById("homehub-logout").addEventListener("click", async (e) => {
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
