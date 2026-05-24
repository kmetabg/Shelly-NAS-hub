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
 *   - inject-ва Account/Logout link в съществуващото menu/nav (не fixed widget)
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
        return;
      }
      document.documentElement.style.visibility = "";
      window.HOMEHUB_AUTH = d;
      injectAccountLinks(d);
    })
    .catch(() => {
      document.documentElement.style.visibility = "";
    });

  /**
   * Inject "Account" + "Logout" в съществуващото menu/nav на страницата,
   * без fixed positioning — за да не припокрива заглавието.
   *
   * Стратегия (опитваме поред, спираме при първи match):
   *   1. <header> .hdr-right            (cameras.html, cameras-config.html)
   *   2. <header> nav, <header> .nav    (recordings.html, faces.html)
   *   3. .header > div:last-child       (events.html — div-based header)
   *   4. <header> directly              (shelly-cam.html и др.)
   *
   * НЕ инжектираме на /account.html (страницата вече показва user info)
   * и на /login.html.
   */
  function injectAccountLinks(authData) {
    if (!authData.auth_enabled || !authData.user) return;
    if (location.pathname === "/account.html") return;
    if (document.getElementById("homehub-account-links")) return;

    // Намираме контейнера за link-овете в реда по-горе.
    const container =
      document.querySelector("header .hdr-right") ||
      document.querySelector("header nav") ||
      document.querySelector("header .nav") ||
      document.querySelector(".header > div:last-child") ||
      document.querySelector("header");

    if (!container) {
      // Няма header → fallback към минимален inline widget в края на body
      // (по-добре от fixed overlap).
      injectFallbackInline(authData);
      return;
    }

    const wrap = document.createElement("span");
    wrap.id = "homehub-account-links";
    wrap.style.cssText =
      "display: inline-flex; align-items: center; gap: 6px; " +
      "margin-left: 6px;";

    const accountLink = document.createElement("a");
    accountLink.href = "/account.html";
    accountLink.title = `Logged in as ${authData.user}`;
    accountLink.textContent = `👤 ${authData.user}`;
    inheritStylesFromSibling(container, accountLink);

    const logoutLink = document.createElement("a");
    logoutLink.href = "#";
    logoutLink.textContent = "Logout";
    logoutLink.title = "Sign out";
    inheritStylesFromSibling(container, logoutLink);
    logoutLink.addEventListener("click", async (e) => {
      e.preventDefault();
      try {
        await fetch("/api/auth/logout", { method: "POST" });
      } catch {}
      location.href = "/login.html";
    });

    wrap.appendChild(accountLink);
    wrap.appendChild(logoutLink);
    container.appendChild(wrap);
  }

  /**
   * Копира class-овете на първия sibling link/бутон в `container`,
   * за да изглежда новият елемент като останалите menu items
   * (back-btn, nav links и т.н.).
   */
  function inheritStylesFromSibling(container, el) {
    const sibling =
      container.querySelector("a, button");
    if (sibling && sibling.className) {
      el.className = sibling.className;
    } else {
      // Минимален fallback стил
      el.style.cssText =
        "color: #58a6ff; text-decoration: none; font-size: 13px;";
    }
  }

  function injectFallbackInline(authData) {
    const bar = document.createElement("div");
    bar.id = "homehub-account-links";
    bar.style.cssText =
      "padding: 6px 12px; background: #161b22; border-bottom: 1px solid #30363d; " +
      "font: 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; " +
      "color: #e6edf3; display: flex; justify-content: flex-end; gap: 12px;";
    bar.innerHTML =
      `<span>👤 ${escapeHtml(authData.user)}</span>` +
      `<a href="/account.html" style="color:#58a6ff;text-decoration:none">Account</a>` +
      `<a href="#" id="homehub-logout-fallback" style="color:#58a6ff;text-decoration:none">Logout</a>`;
    document.body.insertBefore(bar, document.body.firstChild);
    document
      .getElementById("homehub-logout-fallback")
      .addEventListener("click", async (e) => {
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
