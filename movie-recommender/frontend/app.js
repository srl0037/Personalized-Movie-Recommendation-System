const MINIMUM_RATINGS = 5;
const ratingValues = Array.from({ length: 10 }, (_, index) => (index + 1) / 2);
const selectedMovies = new Map();
const palette = [
  ["#0f3857", "#287a91"],
  ["#49304e", "#b07054"],
  ["#173c56", "#598d89"],
  ["#28365e", "#775b91"],
  ["#3c2946", "#c17e48"],
  ["#1e4652", "#68a09a"],
];

const searchInput = document.querySelector("#movie-search");
const searchResults = document.querySelector("#search-results");
const selectedList = document.querySelector("#selected-movies");
const popularGrid = document.querySelector("#popular-movies");
const countBadge = document.querySelector("#rating-count");
const minimumMessage = document.querySelector("#minimum-message");
const minimumNote = document.querySelector(".minimum-note");
const recommendButton = document.querySelector("#recommend-button");
const recommendationsSection = document.querySelector("#recommendations");
const recommendationCards = document.querySelector("#recommendation-cards");
const recommendationStatus = document.querySelector("#recommendation-status");
const posterConfigNote = document.querySelector("#poster-config-note");

let searchTimer;
let searchMovieLookup = new Map();

function escapeText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`The server returned an unexpected response (${response.status}). Check the Flask terminal for details.`);
  }
  if (!response.ok) {
    throw new Error(body.error || "Something went wrong. Please try again.");
  }
  return body;
}

function genreText(movie) {
  return movie.genres.replaceAll("|", " · ");
}

function addMovie(movie) {
  if (selectedMovies.has(movie.movieId)) {
    const existing = document.querySelector(`[data-selected-movie="${movie.movieId}"]`);
    existing?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    existing?.querySelector("select")?.focus();
    return;
  }
  selectedMovies.set(movie.movieId, { ...movie, rating: null });
  renderSelectedMovies();
  document.querySelector(".selected-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  if (window.innerWidth <= 650) {
    document.querySelector(".selected-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function updateRating(movieId, value) {
  const movie = selectedMovies.get(movieId);
  if (!movie) return;
  movie.rating = value === "" ? null : Number(value);
  updateProgress();
}

function renderSelectedMovies() {
  if (selectedMovies.size === 0) {
    selectedList.innerHTML = `
      <div id="empty-state" class="empty-state">
        <span class="empty-icon" aria-hidden="true">＋</span>
        <p>Your picks will show up here.</p>
        <span>Search for a movie or choose a quick add.</span>
      </div>`;
    updateProgress();
    return;
  }

  selectedList.innerHTML = [...selectedMovies.values()].map((movie) => `
    <div class="selected-item" data-selected-movie="${movie.movieId}">
      <div>
        <div class="selected-title" title="${escapeText(movie.title)}">${escapeText(movie.title)}</div>
        <div class="selected-genres">${escapeText(genreText(movie))}</div>
      </div>
      <label class="sr-only" for="rating-${movie.movieId}">Your rating for ${escapeText(movie.title)}</label>
      <select class="rating-select" id="rating-${movie.movieId}" data-rating-id="${movie.movieId}">
        <option value="">Rate movie</option>
        ${ratingValues.map((rating) => `
          <option value="${rating}" ${movie.rating === rating ? "selected" : ""}>
            ${"★".repeat(Math.floor(rating))}${rating % 1 ? "½" : ""} ${rating.toFixed(1)}
          </option>`).join("")}
      </select>
      <button class="remove-rating" type="button" data-remove-id="${movie.movieId}" aria-label="Remove ${escapeText(movie.title)}">×</button>
    </div>`).join("");
  updateProgress();
}

function updateProgress() {
  const ratedCount = [...selectedMovies.values()].filter((movie) => movie.rating !== null).length;
  const remaining = Math.max(0, MINIMUM_RATINGS - ratedCount);
  countBadge.textContent = `${ratedCount} / ${MINIMUM_RATINGS}`;
  recommendButton.disabled = remaining > 0;
  minimumNote.classList.toggle("is-ready", remaining === 0);
  minimumMessage.textContent = remaining
    ? `Rate ${remaining} more ${remaining === 1 ? "movie" : "movies"} to unlock your recommendations.`
    : `${ratedCount} movies rated. You’re ready for recommendations.`;
}

function posterGradient(movieId) {
  const colors = palette[Math.abs(Number(movieId)) % palette.length];
  return `linear-gradient(145deg, ${colors[0]}, ${colors[1]})`;
}

function posterImageMarkup(movie, alt = "") {
  if (!movie.posterUrl) return "";
  try {
    const posterUrl = new URL(movie.posterUrl);
    if (posterUrl.protocol !== "https:" || posterUrl.hostname !== "image.tmdb.org") return "";
    return `<img class="poster-image" src="${escapeText(posterUrl.href)}" alt="${escapeText(alt)}" loading="lazy" decoding="async">`;
  } catch {
    return "";
  }
}

function renderPopularMovies(movies) {
  popularGrid.innerHTML = movies.map((movie) => `
    <article class="popular-item">
      <div class="mini-poster" style="background:${posterGradient(movie.movieId)}" aria-hidden="true">
        ${posterImageMarkup(movie)}
        ${escapeText(movie.title.split(/[,:]/)[0])}
      </div>
      <div class="popular-copy">
        <div class="popular-title" title="${escapeText(movie.title)}">${escapeText(movie.title)}</div>
        <div class="popular-meta">${escapeText(movie.genres.split("|").slice(0, 2).join(" · "))}</div>
      </div>
      <button class="quick-add" type="button" data-add-id="${movie.movieId}" aria-label="Add ${escapeText(movie.title)} to your list">+</button>
    </article>`).join("");
}

function renderSearchResults(movies, query) {
  searchMovieLookup = new Map(movies.map((movie) => [movie.movieId, movie]));
  searchResults.hidden = false;
  searchInput.setAttribute("aria-expanded", "true");
  if (!movies.length) {
    searchResults.innerHTML = `<div class="movie-option"><span class="movie-option-title">No movies found for “${escapeText(query)}”</span></div>`;
    return;
  }
  searchResults.innerHTML = movies.map((movie) => `
    <button class="movie-option" type="button" role="option" data-add-search-id="${movie.movieId}">
      <span class="search-poster" style="background:${posterGradient(movie.movieId)}">${posterImageMarkup(movie)}</span>
      <span>
        <span class="movie-option-title">${escapeText(movie.title)}</span>
        <span class="movie-option-genres">${escapeText(genreText(movie))}</span>
      </span>
      <span class="movie-option-action">${selectedMovies.has(movie.movieId) ? "Added ✓" : "＋ Add"}</span>
    </button>`).join("");
}

function hideSearchResults() {
  searchResults.hidden = true;
  searchInput.setAttribute("aria-expanded", "false");
}

async function searchMovies(query) {
  try {
    const results = await requestJson(`/api/movies/search?q=${encodeURIComponent(query)}`);
    renderSearchResults(results, query);
  } catch (error) {
    searchResults.hidden = false;
    searchInput.setAttribute("aria-expanded", "true");
    searchResults.innerHTML = `<div class="movie-option"><span class="movie-option-title">${escapeText(error.message)}</span></div>`;
  }
}

function renderRecommendations(movies) {
  if (!movies.length) {
    recommendationStatus.textContent = "We couldn’t find any unrated movies to recommend.";
    recommendationCards.replaceChildren();
    return;
  }
  recommendationStatus.textContent = `${movies.length} picks based on your ratings`;
  recommendationCards.innerHTML = movies.map((movie, index) => `
    <article class="movie-card">
      <div class="movie-art" style="--poster-bg:${posterGradient(movie.movieId)}">
        ${posterImageMarkup(movie, `${movie.title} poster`)}
        <span class="movie-art-kicker">MovieMatch pick · ${String(index + 1).padStart(2, "0")}</span>
        <span class="movie-art-title">${escapeText(movie.title)}</span>
      </div>
      <div class="movie-card-info">
        <div class="movie-score">★ ${movie.predictedRating.toFixed(1)}<span>predicted / 5</span></div>
        <p class="movie-genres">${escapeText(genreText(movie))}</p>
      </div>
    </article>`).join("");
  recommendationsSection.hidden = false;
  recommendationsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

searchInput.addEventListener("input", () => {
  window.clearTimeout(searchTimer);
  const query = searchInput.value.trim();
  if (query.length < 2) {
    hideSearchResults();
    return;
  }
  searchTimer = window.setTimeout(() => searchMovies(query), 180);
});

searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideSearchResults();
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".search-control")) hideSearchResults();

  const searchAdd = event.target.closest("[data-add-search-id]");
  const quickAdd = event.target.closest("[data-add-id]");
  const remove = event.target.closest("[data-remove-id]");
  if (searchAdd) {
    const movieId = Number(searchAdd.dataset.addSearchId);
    const movie = searchMovieLookup.get(movieId);
    if (movie) addMovie(movie);
    hideSearchResults();
    searchInput.value = "";
  }
  if (quickAdd) {
    const movie = window.popularMovieLookup?.get(Number(quickAdd.dataset.addId));
    if (movie) addMovie(movie);
  }
  if (remove) {
    selectedMovies.delete(Number(remove.dataset.removeId));
    renderSelectedMovies();
  }
});

document.addEventListener("error", (event) => {
  if (event.target instanceof HTMLImageElement && event.target.classList.contains("poster-image")) {
    event.target.hidden = true;
  }
}, true);

selectedList.addEventListener("change", (event) => {
  const select = event.target.closest("[data-rating-id]");
  if (select) updateRating(Number(select.dataset.ratingId), select.value);
});

recommendButton.addEventListener("click", async () => {
  const ratings = [...selectedMovies.values()]
    .filter((movie) => movie.rating !== null)
    .map(({ movieId, rating }) => ({ movieId, rating }));
  if (ratings.length < MINIMUM_RATINGS) return;

  recommendButton.disabled = true;
  recommendButton.querySelector("span:first-child").textContent = "Finding your picks…";
  recommendationStatus.textContent = "Your taste is on it. Finding ten movies for you…";
  recommendationsSection.hidden = false;
  recommendationsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const movies = await requestJson("/api/recommendations", {
      method: "POST",
      body: JSON.stringify({ ratings }),
    });
    renderRecommendations(movies);
  } catch (error) {
    recommendationCards.replaceChildren();
    recommendationStatus.textContent = error.message;
  } finally {
    recommendButton.querySelector("span:first-child").textContent = "Find my top 10";
    recommendButton.disabled = ratings.length < MINIMUM_RATINGS;
  }
});

document.querySelector("#edit-ratings").addEventListener("click", () => {
  document.querySelector("#rate-movies").scrollIntoView({ behavior: "smooth" });
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    searchInput.focus();
    searchInput.scrollIntoView({ behavior: "smooth", block: "center" });
  }
});

async function initialize() {
  try {
    const health = await requestJson("/api/health");
    posterConfigNote.hidden = health.tmdbConfigured;
  } catch (error) {
    posterConfigNote.hidden = false;
    posterConfigNote.textContent = `Could not check TMDB setup: ${error.message}`;
  }

  try {
    const movies = await requestJson("/api/movies/popular");
    window.popularMovieLookup = new Map(movies.map((movie) => [movie.movieId, movie]));
    renderPopularMovies(movies);
  } catch (error) {
    popularGrid.innerHTML = `<p class="loading-line">${escapeText(error.message)}</p>`;
  }
  updateProgress();
}

initialize();
