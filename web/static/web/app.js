/*
 * Upload, classify, render. No framework and no CDN: the page this replaced
 * pulled AngularJS 1.1.5, jQuery, Bootstrap 3 and a toggle plugin from four
 * CDNs, two of which no longer resolve, so the interface was broken offline
 * and increasingly broken online.
 */
(function () {
  "use strict";

  var form = document.getElementById("upload-form");
  if (!form) return;

  var dropzone = document.getElementById("dropzone");
  var input = document.getElementById("file-input");
  var progress = document.getElementById("progress");
  var errorBox = document.getElementById("error");
  var errorText = document.getElementById("error-text");
  var result = document.getElementById("result");
  var historyList = document.getElementById("history");

  var SEQ_STEPS = 5;

  function csrfToken() {
    var field = form.querySelector("[name=csrfmiddlewaretoken]");
    return field ? field.value : "";
  }

  function percent(value) {
    return Math.round(value * 100) + "%";
  }

  function clock(seconds) {
    var whole = Math.round(seconds);
    return Math.floor(whole / 60) + ":" + String(whole % 60).padStart(2, "0");
  }

  /* Map a probability onto the sequential ramp. Five steps, because past
   * about seven bins adjacent classes stop being distinguishable. */
  function seqVar(probability) {
    var step = Math.min(SEQ_STEPS, Math.max(1, Math.ceil(probability * SEQ_STEPS)));
    return "var(--seq-" + step + ")";
  }

  function showError(message) {
    errorText.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.hidden = true;
  }

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderBars(ranked) {
    var host = document.getElementById("bars");
    host.replaceChildren();
    var max = ranked.reduce(function (acc, item) {
      return Math.max(acc, item.probability);
    }, 0) || 1;

    ranked.forEach(function (item) {
      var row = element("div", "bar-row");
      row.appendChild(element("span", "bar-label", item.genre));

      var track = element("div", "bar-track");
      var fill = element("div", "bar-fill");
      fill.style.width = Math.max(1, (item.probability / max) * 100) + "%";
      fill.title = item.genre + ": " + percent(item.probability);
      track.appendChild(fill);
      row.appendChild(track);

      row.appendChild(element("span", "bar-value", item.probability.toFixed(2)));
      host.appendChild(row);
    });
  }

  function renderTimeline(segments, duration) {
    var block = document.getElementById("timeline-block");
    var host = document.getElementById("timeline");
    host.replaceChildren();

    if (!segments.length) {
      block.hidden = true;
      return;
    }
    block.hidden = false;

    segments.forEach(function (segment) {
      var cell = element("div", "tl-seg");
      var bar = element("div", "tl-bar");
      bar.style.background = seqVar(segment.probability);
      cell.appendChild(bar);
      cell.appendChild(element("span", "tl-cap", segment.genre));
      cell.title =
        clock(segment.start) + "–" + clock(segment.end) + "  " +
        segment.genre + " " + percent(segment.probability) +
        (segment.runner_up ? "  (then " + segment.runner_up + ")" : "");
      host.appendChild(cell);
    });
    document.getElementById("tl-end").textContent = clock(duration);
  }

  function renderNeighbours(neighbours) {
    var host = document.getElementById("neighbours");
    host.replaceChildren();

    neighbours.forEach(function (neighbour) {
      var row = element("li", "nb-row");
      var name = element("span", "nb-track");
      name.appendChild(document.createTextNode(neighbour.track));
      row.appendChild(name);

      var track = element("span", "nb-track-bar");
      var fill = element("span", "nb-fill");
      fill.style.width = Math.max(1, neighbour.closer_than * 100) + "%";
      track.appendChild(fill);
      row.appendChild(track);

      var value = element("span", "nb-value", percent(neighbour.closer_than));
      value.title = "closer than " + percent(neighbour.closer_than) +
        " of GTZAN track pairs (distance " + neighbour.distance + ")";
      row.appendChild(value);
      host.appendChild(row);
    });
  }

  function renderResult(data) {
    var verdict = data.verdict;

    document.getElementById("v-genre").textContent = verdict.genre;
    document.getElementById("v-prob").textContent =
      percent(verdict.probability) + " · " + percent(verdict.margin) + " clear of the runner-up";

    var chip = document.getElementById("v-chip");
    chip.className = "chip is-" + verdict.band;
    document.getElementById("v-band").textContent = verdict.band;
    document.getElementById("v-reason").textContent = verdict.reason;

    var meta = document.getElementById("v-meta");
    meta.replaceChildren();
    [
      data.name,
      clock(data.duration_seconds) + " scored",
      data.sample_rate + " Hz mono",
      data.extractor,
      (data.cached ? "cache hit" : data.elapsed_ms + " ms")
    ].forEach(function (text) {
      meta.appendChild(element("span", null, text));
    });

    renderBars(data.ranked);
    renderTimeline(data.timeline || [], data.duration_seconds);
    renderNeighbours(data.neighbours || []);

    result.hidden = false;
  }

  function prependHistory(data) {
    var existing = historyList.querySelector('[data-id="' + data.id + '"]');
    if (existing) existing.remove();

    var empty = historyList.querySelector(".empty");
    if (empty) empty.closest("li").remove();

    var item = element("li");
    item.dataset.id = data.id;

    var head = element("div", "h-head");
    var name = element("span", "h-name", data.name);
    name.title = data.name;
    head.appendChild(name);
    head.appendChild(element("span", "h-genre", data.verdict.genre));
    item.appendChild(head);

    var sub = element("div", "h-sub");
    var bar = element("span", "h-bar");
    var fill = element("i");
    fill.style.width = Math.max(1, data.verdict.probability * 100) + "%";
    bar.appendChild(fill);
    sub.appendChild(bar);
    sub.appendChild(element("span", "nb-value", data.verdict.probability.toFixed(2)));

    var remove = element("button", "btn-link", "remove");
    remove.dataset.delete = data.id;
    sub.appendChild(remove);
    item.appendChild(sub);

    if (data.audio_url) {
      var player = document.createElement("audio");
      player.controls = true;
      player.preload = "none";
      player.src = data.audio_url;
      item.appendChild(player);
    }

    historyList.prepend(item);
    while (historyList.children.length > 20) historyList.lastElementChild.remove();
  }

  function upload(file) {
    if (!file) return;
    clearError();
    progress.hidden = false;
    dropzone.classList.remove("is-over");

    var body = new FormData();
    body.append("file", file);

    fetch("/api/classify/", {
      method: "POST",
      body: body,
      headers: { "X-CSRFToken": csrfToken() }
    })
      .then(function (response) {
        return response.json().then(function (data) {
          if (!response.ok) throw new Error(data.error || "classification failed");
          return data;
        });
      })
      .then(function (data) {
        renderResult(data);
        prependHistory(data);
      })
      .catch(function (err) {
        showError(err.message);
      })
      .finally(function () {
        progress.hidden = true;
        input.value = "";
      });
  }

  dropzone.addEventListener("click", function () { input.click(); });
  dropzone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });

  ["dragenter", "dragover"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.add("is-over");
    });
  });

  ["dragleave", "drop"].forEach(function (name) {
    dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      dropzone.classList.remove("is-over");
    });
  });

  dropzone.addEventListener("drop", function (event) {
    upload(event.dataTransfer.files[0]);
  });

  input.addEventListener("change", function () { upload(input.files[0]); });
  form.addEventListener("submit", function (event) { event.preventDefault(); });

  historyList.addEventListener("click", function (event) {
    var button = event.target.closest("[data-delete]");
    if (!button) return;
    var id = button.dataset.delete;
    fetch("/api/analyses/" + id + "/", {
      method: "DELETE",
      headers: { "X-CSRFToken": csrfToken() }
    }).then(function () {
      var row = historyList.querySelector('[data-id="' + id + '"]');
      if (row) row.remove();
      if (!historyList.children.length) {
        var placeholder = element("li");
        placeholder.appendChild(element("p", "empty", "Nothing classified yet. Upload a track to start."));
        historyList.appendChild(placeholder);
      }
    });
  });
})();
