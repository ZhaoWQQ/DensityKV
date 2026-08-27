const menuButton = document.querySelector("[data-menu-button]");
const mobileNav = document.querySelector("[data-mobile-nav]");

menuButton?.addEventListener("click", () => {
  const isOpen = mobileNav?.classList.toggle("is-open") ?? false;
  menuButton.setAttribute("aria-expanded", String(isOpen));
  menuButton.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
});

mobileNav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    mobileNav.classList.remove("is-open");
    menuButton?.setAttribute("aria-expanded", "false");
    menuButton?.setAttribute("aria-label", "Open navigation");
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !mobileNav?.classList.contains("is-open")) return;
  mobileNav.classList.remove("is-open");
  menuButton?.setAttribute("aria-expanded", "false");
  menuButton?.setAttribute("aria-label", "Open navigation");
  menuButton?.focus();
});

window.addEventListener("resize", () => {
  if (window.innerWidth <= 820 || !mobileNav?.classList.contains("is-open")) return;
  mobileNav.classList.remove("is-open");
  menuButton?.setAttribute("aria-expanded", "false");
  menuButton?.setAttribute("aria-label", "Open navigation");
});

const copyText = async (button, selector, copiedLabel, fallbackLabel) => {
  const value = document.querySelector(selector)?.textContent?.trim();
  if (!value) return;

  try {
    await navigator.clipboard.writeText(value);
    const icon = button.querySelector("svg");
    button.title = copiedLabel;
    button.setAttribute("aria-label", copiedLabel);
    icon?.setAttribute("data-lucide", "check");
    window.lucide?.createIcons({ "stroke-width": 1.8 });
    window.setTimeout(() => {
      button.title = fallbackLabel;
      button.setAttribute("aria-label", fallbackLabel);
      button.querySelector("svg")?.setAttribute("data-lucide", "copy");
      window.lucide?.createIcons({ "stroke-width": 1.8 });
    }, 1800);
  } catch {
    button.title = "Select the text to copy";
  }
};

document.querySelector("[data-copy-command]")?.addEventListener("click", (event) => {
  copyText(event.currentTarget, "[data-command]", "Command copied", "Copy command");
});

document.querySelector("[data-copy-bibtex]")?.addEventListener("click", (event) => {
  copyText(event.currentTarget, "[data-bibtex]", "BibTeX copied", "Copy BibTeX");
});

if (window.lucide) {
  window.lucide.createIcons({ "stroke-width": 1.8 });
}
