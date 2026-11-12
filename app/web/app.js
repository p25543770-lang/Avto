/* AI-режиссёр — клиентская логика интерфейса */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    file: null,
    jobId: null,
    events: null,
    currentStep: 0,
  };

  /* ---------------- утилиты ---------------- */
  function toast(message, kind = "") {
    const box = $("toast");
    box.textContent = message;
    box.className = "toast show " + kind;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => (box.className = "toast " + kind), 4200);
  }

  function log(text, kind = "") {
    const console_ = $("console");
    const line = document.createElement("div");
    line.className = "line " + kind;
    const time = document.createElement("span");
    time.className = "t";
    time.textContent = new Date().toLocaleTimeString("ru-RU", { hour12: false });
    line.appendChild(time);
    line.appendChild(document.createTextNode(text));
    console_.appendChild(line);
    console_.scrollTop = console_.scrollHeight;
  }

  function fmtBytes(bytes) {
    if (!bytes) return "0 Б";
    const units = ["Б", "КБ", "МБ", "ГБ"];
    const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
    return (bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0) + " " + units[index];
  }

  /* ---------------- системная информация ---------------- */
  async function loadSystem() {
    const chips = $("systemChips");
    try {
      const data = await (await fetch("/api/system")).json();
      chips.innerHTML = "";

      const chip = (text, kind) => {
        const span = document.createElement("span");
        span.className = "chip " + (kind || "");
        span.textContent = text;
        chips.appendChild(span);
      };

      chip(
        data.ffmpeg.available ? `ffmpeg ✓ ${data.encoder}` : "ffmpeg не найден",
        data.ffmpeg.available ? (data.encoder === "h264_nvenc" ? "ok" : "warn") : "bad"
      );
      chip(
        data.gpu.cuda ? `GPU: ${data.gpu.name}` : `GPU: ${data.gpu.name} (CPU-режим)`,
        data.gpu.cuda ? "ok" : "warn"
      );
      chip(
        data.whisper.available ? `whisper ✓ ${data.whisper.model}` : "faster-whisper не установлен",
        data.whisper.available ? "ok" : "bad"
      );
      chip(
        data.llm.available ? `LLM ✓ ${data.llm.model}` : "LLM недоступен",
        data.llm.available ? "ok" : "warn"
      );

      $("pathsInfo").textContent = `input → ${data.paths.input} · output → ${data.paths.output}`;
      $("watchToggle").checked = !!data.watch;
    } catch (error) {
      chips.innerHTML = '<span class="chip bad">сервер недоступен</span>';
    }
  }

  async function loadSettings() {
    try {
      const data = await (await fetch("/api/settings")).json();
      $("llmUrl").value = data.llm_base_url || "";
      $("llmModel").value = data.llm_model || "";
      $("llmKey").value = data.llm_api_key && data.llm_api_key !== "***" ? data.llm_api_key : "";
    } catch (error) {
      /* не критично */
    }
  }

  /* ---------------- загрузка файла ---------------- */
  const dropzone = $("dropzone");
  const fileInput = $("fileInput");

  $("pickBtn").addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
  });
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") fileInput.click();
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) setFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.remove("drag");
    })
  );
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setFile(file);
  });

  function setFile(file) {
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|mkv|avi|webm|m4v)$/i.test(file.name)) {
      toast("Это не видеофайл", "error");
      return;
    }
    state.file = file;
    dropzone.querySelector(".dz-title").textContent = file.name;
    dropzone.querySelector(".dz-sub").innerHTML = `${fmtBytes(file.size)} — <span class="link-btn" id="resetFile">выбрать другой</span>`;
    const reset = $("resetFile");
    if (reset)
      reset.addEventListener("click", (event) => {
        event.stopPropagation();
        state.file = null;
        fileInput.value = "";
        dropzone.querySelector(".dz-title").textContent = "Перетащите видео сюда";
        dropzone.querySelector(".dz-sub").innerHTML = 'или <button type="button" class="link-btn" id="pickBtn">выберите файл</button>';
        $("pickBtn").addEventListener("click", (event) => {
          event.stopPropagation();
          fileInput.click();
        });
        $("startBtn").disabled = true;
      });
    $("startBtn").disabled = false;
    log(`Файл выбран: ${file.name} (${fmtBytes(file.size)})`);
  }

  /* ---------------- опции ---------------- */
  function collectOptions() {
    const checked = (id) =>
      Array.from($(id).querySelectorAll("input:checked")).map((input) => input.value);
    return {
      final_formats: checked("finalFormats"),
      short_formats: checked("shortFormats"),
      short_count: Number($("shortCount").value),
      short_min_duration: Number($("shortMin").value),
      short_max_duration: Number($("shortMax").value),
      language: $("language").value,
      whisper_model: $("whisperModel").value,
      encoder: $("encoder").value,
      silence_margin: Number($("silenceMargin").value),
      remove_duplicates: $("removeDuplicates").checked,
      use_llm: $("useLlm").checked,
    };
  }

  /* ---------------- запуск задачи ---------------- */
  $("startBtn").addEventListener("click", startJob);

  async function startJob() {
    if (!state.file) {
      toast("Сначала выберите видео", "error");
      return;
    }
    const options = collectOptions();
    if (!options.final_formats.length) {
      toast("Выберите хотя бы один формат чистовика", "error");
      return;
    }
    if (!options.short_formats.length) {
      toast("Выберите формат шортсов или уменьшите их количество", "error");
      return;
    }

    const form = new FormData();
    form.append("file", state.file);
    form.append("options", JSON.stringify(options));

    $("startBtn").disabled = true;
    $("cancelBtn").hidden = false;
    $("results").classList.add("hidden");
    $("console").innerHTML = "";
    resetSteps();

    try {
      const response = await fetch("/api/jobs", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Ошибка загрузки");
      state.jobId = data.job_id;
      log(`Задача запущена: ${data.source}`);
      subscribe(state.jobId);
    } catch (error) {
      toast(error.message, "error");
      log(`Ошибка: ${error.message}`, "err");
      $("startBtn").disabled = false;
      $("cancelBtn").hidden = true;
    }
  }

  $("cancelBtn").addEventListener("click", async () => {
    if (!state.jobId) return;
    await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
    log("Остановка задачи…", "err");
  });

  /* ---------------- SSE ---------------- */
  function subscribe(jobId) {
    if (state.events) state.events.close();
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    state.events = source;

    source.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      (data.logs || []).forEach((item) => log(item.text));
      updateProgress(data.progress, data.message, data.status);
      if (data.status === "done") {
        source.close();
        state.events = null;
        await showResults(jobId);
        $("startBtn").disabled = false;
        $("cancelBtn").hidden = true;
        toast("Готово! Файлы в папке output/", "ok");
      } else if (data.status === "error") {
        source.close();
        state.events = null;
        $("startBtn").disabled = false;
        $("cancelBtn").hidden = true;
        toast(data.error || "Ошибка монтажа", "error");
      } else if (data.status === "canceled") {
        source.close();
        state.events = null;
        $("startBtn").disabled = false;
        $("cancelBtn").hidden = true;
        toast("Задача остановлена", "error");
      }
    };
    source.onerror = () => {
      /* поток закрывается сам при завершении задачи */
    };
  }

  function updateProgress(value, message, status) {
    const pct = Math.max(0, Math.min(100, Number(value) || 0));
    $("progressBar").style.width = pct + "%";
    $("progressPct").textContent = Math.round(pct) + "%";
    if (message) $("progressText").textContent = message;
    updateSteps(pct, status);
  }

  const STEP_RANGES = [
    [0, 5],
    [5, 20],
    [20, 38],
    [38, 50],
    [50, 65],
    [65, 90],
    [90, 101],
  ];

  function updateSteps(pct, status) {
    let current = 0;
    STEP_RANGES.forEach(([from, to], index) => {
      if (pct >= from && pct < to) current = index;
    });
    if (pct >= 100) current = 6;

    document.querySelectorAll(".step").forEach((step, index) => {
      const number = index + 1;
      const state_ = step.querySelector(".step-state");
      step.classList.remove("active", "done");
      state_.textContent = "";
      if (number < current || (status === "done" && number <= 6)) {
        step.classList.add("done");
        state_.textContent = "✓";
      } else if (number === current && status !== "done") {
        step.classList.add("active");
        state_.textContent = "…";
      }
    });
    state.currentStep = current;
  }

  function resetSteps() {
    document.querySelectorAll(".step").forEach((step) => {
      step.classList.remove("active", "done");
      step.querySelector(".step-state").textContent = "";
    });
    $("progressBar").style.width = "0%";
    $("progressPct").textContent = "0%";
    $("progressText").textContent = "Загрузка файла…";
  }

  /* ---------------- результаты ---------------- */
  async function showResults(jobId) {
    const response = await fetch(`/api/jobs/${jobId}/result`);
    const data = await response.json();
    const result = data.result || {};
    $("results").classList.remove("hidden");

    const finalList = $("finalList");
    finalList.innerHTML = "";
    (result.final || []).forEach((item) => {
      const card = document.createElement("div");
      card.className = "video-card";
      card.innerHTML = `
        <video src="/api/files/${encodeURIComponent(item.filename)}" controls preload="metadata"></video>
        <div class="video-meta">
          <div>
            <div class="name">${item.filename}</div>
            <div class="sub">${item.width}×${item.height} · ${item.duration} сек · ${fmtBytes(item.size)}</div>
          </div>
          <span class="badge ${item.aspect === "9:16" ? "vertical" : ""}">${item.aspect}</span>
          <a class="dl" href="/api/files/${encodeURIComponent(item.filename)}" download>Скачать</a>
        </div>`;
      finalList.appendChild(card);
    });

    const shortList = $("shortList");
    shortList.innerHTML = "";
    (result.shorts || []).forEach((item) => {
      const card = document.createElement("div");
      card.className = "short-card" + (item.aspect === "16:9" ? " landscape" : "");
      card.innerHTML = `
        <video src="/api/files/${encodeURIComponent(item.filename)}" controls preload="metadata"></video>
        <div class="short-info">
          <div class="short-title">${item.title || "Шортс"}</div>
          <div class="short-sub">${item.aspect} · ${item.duration} сек · <a class="dl" href="/api/files/${encodeURIComponent(item.filename)}" download>скачать</a></div>
        </div>`;
      shortList.appendChild(card);
    });

    const chaptersBox = $("chaptersBox");
    const chapters = result.chapters || [];
    if (chapters.length) {
      chaptersBox.classList.remove("hidden");
      $("chaptersList").innerHTML = chapters
        .map((chapter) => `<li><code>${chapter.time}</code> ${chapter.title}</li>`)
        .join("");
    } else {
      chaptersBox.classList.add("hidden");
    }

    const downloads = $("downloads");
    downloads.innerHTML = "";
    const files = [
      ["Таймкоды глав", result.chapters_filename],
      ["Расшифровка", result.transcript_filename],
    ].filter((item) => item[1]);
    files.forEach(([label, filename]) => {
      const link = document.createElement("a");
      link.className = "dl";
      link.href = "/api/files/" + encodeURIComponent(filename);
      link.download = filename;
      link.textContent = `↓ ${label}`;
      downloads.appendChild(link);
    });
    downloads.classList.toggle("hidden", !files.length);
  }

  /* ---------------- настройки LLM ---------------- */
  $("saveLlm").addEventListener("click", async () => {
    const payload = {
      llm_base_url: $("llmUrl").value.trim(),
      llm_model: $("llmModel").value.trim(),
    };
    const key = $("llmKey").value.trim();
    if (key) payload.llm_api_key = key;

    try {
      await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      toast("Настройки сохранены", "ok");
      loadSystem();
    } catch (error) {
      toast("Не удалось сохранить настройки", "error");
    }
  });

  /* ---------------- слежение за папкой ---------------- */
  $("watchToggle").addEventListener("change", async (event) => {
    const enabled = event.target.checked;
    try {
      await fetch("/api/watch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      toast(enabled ? "Слежение за папкой input/ включено" : "Слежение выключено", "ok");
    } catch (error) {
      toast("Ошибка переключения слежения", "error");
    }
  });

  /* ---------------- старт ---------------- */
  loadSystem();
  loadSettings();
  setInterval(loadSystem, 30000);
})();
