document.addEventListener("DOMContentLoaded", () => {
  fetch("./descriptions/artworks.json")
    .then((res) => res.json())
    .then((artworks) => {
      const gallery = document.getElementById("gallery");
      artworks.forEach(({ title, image, description }) => {
        const el = document.createElement("div");
        el.className = "artwork";
        el.innerHTML = `
          <img src="${image}" alt="${title}" />
          <h2>${title}</h2>
          <p>${description}</p>
        `;
        gallery.appendChild(el);
      });
    })
    .catch(() => {
      document.getElementById("gallery").innerHTML =
        "<p>⚠️ No artwork found yet. Add some to descriptions/artworks.json</p>";
    });
});
