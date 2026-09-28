window.riverPlastNavigationLoaded = true;

function showRiverPlastPage(value) {
  const nav = document.getElementById("main_nav");
  const link = nav && nav.querySelector(`a[data-value="${value}"]`);
  if (!link) return;

  const pane = document.querySelector(link.getAttribute("href"));
  if (!pane) return;

  nav.querySelectorAll(".nav-link").forEach((item) => {
    item.classList.remove("active");
    item.setAttribute("aria-selected", "false");
  });
  pane.parentElement.querySelectorAll(":scope > .tab-pane").forEach((item) => {
    item.classList.remove("active");
  });
  link.classList.add("active");
  link.setAttribute("aria-selected", "true");
  pane.classList.add("active");
}

document.addEventListener("click", function (event) {
  const control = event.target.closest("[data-river-plast-nav]")
    || event.target.closest("#main_nav a.nav-link");
  if (!control) return;

  const value = control.dataset.riverPlastNav || control.dataset.value;
  if (!["home", "calculator", "explorer", "methods"].includes(value)) return;

  event.preventDefault();
  showRiverPlastPage(value);
  history.replaceState(null, "", location.pathname);
}, true);
