/**
 * SmartSafety Vision — script.js
 * Gestion UI : flux vidéo, uploads, système de toasts, overlay CCTV.
 */

"use strict";

/* ════════════════════════════════════════════
   RÉFÉRENCES DOM
════════════════════════════════════════════ */
const cameraButton = document.getElementById("cameraButton");
const demoButton = document.getElementById("demoButton");
const uploadForm = document.getElementById("uploadForm");
const modelForm = document.getElementById("modelForm");
const videoFileInput = document.getElementById("videoFile");
const modelFileInput = document.getElementById("modelFile");
const videoFileName = document.getElementById("videoFileName");
const modelFileName = document.getElementById("modelFileName");
const statusBox = document.getElementById("statusBox");
const videoStream = document.getElementById("videoStream");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const liveBadge = document.getElementById("liveBadge");
const overlayTime = document.getElementById("overlayTime");
const toastContainer = document.getElementById("toastContainer");
const stopStreamBtn = document.getElementById("stopStreamBtn");
const modelStatusText = document.getElementById("modelStatusText");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const themeIcon = document.getElementById("themeIcon");

/* ════════════════════════════════════════════
   ICÔNES TOAST PAR TYPE
════════════════════════════════════════════ */
const TOAST_ICONS = {
  success: "fa-circle-check",
  danger: "fa-circle-xmark",
  warning: "fa-triangle-exclamation",
  info: "fa-circle-info",
};

const TOAST_LABELS = {
  success: "Succès",
  danger: "Erreur",
  warning: "Attention",
  info: "Info",
};

/* ════════════════════════════════════════════
   SYSTÈME DE TOASTS
════════════════════════════════════════════ */
/**
 * Affiche un toast de notification flottant.
 * @param {string} message  - Message principal
 * @param {"success"|"danger"|"warning"|"info"} type
 * @param {number} duration - Durée en ms (0 = permanent)
 */
function showToast(message, type = "info", duration = 4500) {
  const icon = TOAST_ICONS[type] || TOAST_ICONS.info;
  const label = TOAST_LABELS[type] || "Info";

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.setAttribute("role", "alert");
  toast.innerHTML = `
    <div class="toast-icon"><i class="fas ${icon}"></i></div>
    <div class="toast-body">
      <p class="toast-title">${label}</p>
      <p class="toast-msg">${escapeHtml(message)}</p>
    </div>
    <button class="toast-close" aria-label="Fermer"><i class="fas fa-xmark"></i></button>
  `;

  const closeBtn = toast.querySelector(".toast-close");
  const dismiss = () => {
    toast.classList.add("toast-exit");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  };

  closeBtn.addEventListener("click", dismiss);
  if (duration > 0) setTimeout(dismiss, duration);

  toastContainer.appendChild(toast);
}

/** Échappe le HTML pour éviter les injections. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ════════════════════════════════════════════
   STATUT RAPIDE (SIDEBAR)
════════════════════════════════════════════ */
/**
 * Met à jour la carte "Statut système" dans la sidebar.
 * @param {string} message
 * @param {"success"|"danger"|"warning"|"info"} type
 */
function setStatus(message, type = "info") {
  const iconMap = {
    success: "fa-circle-check",
    danger: "fa-circle-xmark",
    warning: "fa-triangle-exclamation",
    info: "fa-circle-info",
  };
  const icon = iconMap[type] || iconMap.info;
  statusBox.innerHTML = `
    <div class="status-row">
      <i class="fas ${icon} status-icon ${type}"></i>
      <span>${escapeHtml(message)}</span>
    </div>`;
}

/* ════════════════════════════════════════════
   HORLOGE TEMPS RÉEL — OVERLAY CCTV
════════════════════════════════════════════ */
function updateOverlayClock() {
  const now = new Date();
  overlayTime.textContent = now.toLocaleTimeString("fr-FR", {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}
setInterval(updateOverlayClock, 1000);
updateOverlayClock();

/* ════════════════════════════════════════════
   GESTION DU FLUX VIDÉO
════════════════════════════════════════════ */
/**
 * Active le flux MJPEG et masque le placeholder.
 */
function activateStream() {
  videoStream.src = "/video_feed?" + Date.now(); // cache-bust
  videoPlaceholder.classList.add("hidden");
  liveBadge.classList.add("active");
  if (stopStreamBtn) {
    stopStreamBtn.disabled = false;
  }
  startAlertPolling();
}

function deactivateStream() {
  if (stopStreamBtn) {
    stopStreamBtn.disabled = true;
  }
  videoStream.src = "";
  videoPlaceholder.classList.remove("hidden");
  liveBadge.classList.remove("active");
  stopAlertPolling();
}

async function stopStream() {
  try {
    await fetch("/stop_stream", { method: "POST" });
    setStatus("Flux vidéo arrêté.", "info");
    showToast("Flux vidéo arrêté.", "info");
  } catch (err) {
    setStatus("Impossible d'arrêter le flux.", "danger");
    showToast("Impossible d'arrêter le flux.", "danger");
  } finally {
    deactivateStream();
  }
}

/**
 * Appelle un endpoint POST et active le flux si succès.
 * @param {string} endpoint
 * @param {string} successMsg
 */
async function startStream(endpoint, successMsg) {
  setStatus("Connexion en cours…", "info");
  try {
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Erreur lors de l'initialisation du flux.");
    }

    activateStream();
    setStatus(successMsg, "success");
    showToast(successMsg, "success");
  } catch (err) {
    setStatus(err.message, "danger");
    showToast(err.message, "danger");
  }
}

/* ════════════════════════════════════════════
   BOUTONS SOURCES
════════════════════════════════════════════ */
cameraButton.addEventListener("click", () =>
  startStream("/start_camera", "Flux caméra locale actif.")
);

demoButton.addEventListener("click", () =>
  startStream("/start_demo", "Vidéo de démonstration lancée.")
);

if (stopStreamBtn) {
  stopStreamBtn.addEventListener("click", stopStream);
}

/* ════════════════════════════════════════════
   GESTION DES FICHIERS — AFFICHAGE DU NOM
════════════════════════════════════════════ */
/**
 * Met à jour l'affichage du fichier sélectionné.
 * @param {HTMLInputElement} input
 * @param {HTMLElement}      display
 * @param {"video"|"model"}  kind
 */
function updateFileDisplay(input, display, kind) {
  if (!input.files || input.files.length === 0) {
    display.classList.remove("has-file");
    display.innerHTML = `<i class="fas fa-file-circle-xmark file-icon-none"></i><span>${kind === "video" ? "Aucun fichier sélectionné" : "Aucun modèle sélectionné"
      }</span>`;
    return;
  }
  const name = input.files[0].name;
  display.classList.add("has-file");
  display.innerHTML = `<i class="fas fa-file-circle-check"></i><span>${escapeHtml(name)}</span>`;
}

videoFileInput.addEventListener("change", () =>
  updateFileDisplay(videoFileInput, videoFileName, "video")
);
modelFileInput.addEventListener("change", () =>
  updateFileDisplay(modelFileInput, modelFileName, "model")
);

/* ════════════════════════════════════════════
   DRAG & DROP — ZONES DE DÉPÔT
════════════════════════════════════════════ */
function setupDragDrop(dropZone, fileInput, display, kind) {
  ["dragenter", "dragover"].forEach(evt =>
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    })
  );

  ["dragleave", "drop"].forEach(evt =>
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    })
  );

  dropZone.addEventListener("drop", e => {
    const files = e.dataTransfer.files;
    if (!files.length) return;

    // Transférer dans l'input natif (pour FormData)
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    fileInput.files = dt.files;
    updateFileDisplay(fileInput, display, kind);
  });
}

setupDragDrop(
  document.getElementById("videoDropZone"),
  videoFileInput, videoFileName, "video"
);
setupDragDrop(
  document.getElementById("modelDropZone"),
  modelFileInput, modelFileName, "model"
);

/* ════════════════════════════════════════════
   UPLOAD VIDÉO MP4
════════════════════════════════════════════ */
uploadForm.addEventListener("submit", async e => {
  e.preventDefault();

  if (!videoFileInput.files.length) {
    showToast("Sélectionnez un fichier MP4 avant d'uploader.", "warning");
    return;
  }

  const file = videoFileInput.files[0];
  if (file.type !== "video/mp4") {
    showToast("Seuls les fichiers MP4 sont acceptés.", "warning");
    return;
  }

  setStatus("Upload en cours…", "info");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload_video", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Erreur pendant l'upload vidéo.");

    activateStream();
    setStatus(data.message, "success");
    showToast(data.message, "success");
  } catch (err) {
    setStatus(err.message, "danger");
    showToast(err.message, "danger");
  }
});

/* ════════════════════════════════════════════
   UPLOAD MODÈLE .PT
════════════════════════════════════════════ */
modelForm.addEventListener("submit", async e => {
  e.preventDefault();

  if (!modelFileInput.files.length) {
    showToast("Sélectionnez un fichier .pt avant de charger le modèle.", "warning");
    return;
  }

  const file = modelFileInput.files[0];
  if (!file.name.toLowerCase().endsWith(".pt")) {
    showToast("Le modèle doit être un fichier .pt (Ultralytics YOLOv8).", "warning");
    return;
  }

  // Validate file size (max 500MB)
  const maxSize = 500 * 1024 * 1024;
  if (file.size > maxSize) {
    showToast(`Le fichier est trop volumineux (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum 500MB.`, "warning");
    return;
  }

  setStatus("Chargement du modèle…", "info");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload_model", { method: "POST", body: formData });
    
    if (!res.ok) {
      let errorDetail = "Erreur pendant le chargement du modèle.";
      try {
        const errorData = await res.json();
        errorDetail = errorData.detail || errorData.message || errorDetail;
      } catch (e) {
        // Si la réponse n'est pas JSON, utilise le texte d'état HTTP
        errorDetail = `Erreur ${res.status}: ${res.statusText}`;
      }
      throw new Error(errorDetail);
    }

    const data = await res.json();

    const msg = `Modèle chargé : ${data.model_name}`;
    const allOk = data.person_detected && data.epi_classes_available;
    setStatus(msg, allOk ? "success" : "warning");
    showToast(msg, allOk ? "success" : "warning");

    // Affiche chaque avertissement détaillé retourné par le backend
    if (data.warnings && data.warnings.length > 0) {
      for (const warn of data.warnings) {
        showToast(warn, "warning", 10000);
      }
    }

    // Info des classes détectées (toast info)
    if (data.person_detected) {
      showToast(
        `Classe personne : ${data.person_classes.join(", ")} · ` +
        (data.epi_classes_available
          ? `EPI : ${data.epi_classes.join(", ")}`
          : "Pas de classe EPI dans ce modèle"),
        data.epi_classes_available ? "success" : "warning",
        7000
      );
    }

    // Reset form
    modelFileInput.value = "";
    updateFileDisplay(modelFileInput, modelFileName, "model");

  } catch (err) {
    console.error("[SmartSafety] Erreur upload modèle:", err);
    setStatus(err.message, "danger");
    showToast(err.message, "danger");
  }
});

/* ════════════════════════════════════════════
   GESTION ERREUR FLUX VIDÉO
════════════════════════════════════════════ */
videoStream.addEventListener("error", () => {
  if (!videoStream.src || videoStream.src === window.location.href) return;
  showToast("Le flux vidéo s'est interrompu.", "warning");
  deactivateStream();
});

async function fetchModelInfo() {
  try {
    const res = await fetch("/model_info");
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Impossible de lire les infos du modèle.");
    }

    if (!data.epi_classes_available) {
      setStatus("Modèle chargé sans classes EPI.", "danger");
      if (modelStatusText) {
        modelStatusText.textContent = "Modèle chargé sans classes EPI.";
      }
      showToast("Le modèle actuel ne reconnaît pas d'EPI.", "warning", 9000);
    } else if (!data.person_detected) {
      setStatus("Modèle EPI-only chargé.", "warning");
      if (modelStatusText) {
        modelStatusText.textContent = "Modèle EPI-only : détection d'équipement sans personne.";
      }
      showToast("Modèle EPI-only détecté : affichage des EPI sans association à une personne.", "warning", 9000);
    } else {
      setStatus("Modèle EPI prêt.", "success");
      if (modelStatusText) {
        modelStatusText.textContent = `Personne : ${data.person_classes.join(", ")} · EPI : ${data.epi_classes.join(", ")}`;
      }
    }
  } catch (err) {
    console.error(err);
    if (modelStatusText) {
      modelStatusText.textContent = "Impossible de récupérer les infos du modèle.";
    }
  }
}

/* ════════════════════════════════════════════
   ALERTE SONORE POUR VIOLATIONS EPI
════════════════════════════════════════════ */
let alertInterval = null;

/**
 * Joue un bip simple via Web Audio API.
 */
function playBeep() {
  try {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
    oscillator.type = 'square';
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.2);
    console.log("[🔊 BEEEP] Son joué avec succès!");
  } catch (err) {
    console.warn("Impossible de jouer le bip :", err);
  }
}

/**
 * Vérifie l'état d'alerte toutes les secondes.
 */
async function checkAlert() {
  try {
    const res = await fetch("/alert");
    if (!res.ok) return;
    const data = await res.json();
    if (data.alert) {
      console.log("[🔔 VIOLATION DÉTECTÉE] EPI manquant(s) détecté(s) — Déclenchement alerte sonore.");
      playBeep();
    }
  } catch (err) {
    console.warn("Erreur vérification alerte :", err);
  }
}

/**
 * Démarre la vérification d'alerte.
 */
function startAlertPolling() {
  if (alertInterval) clearInterval(alertInterval);
  alertInterval = setInterval(checkAlert, 1000);
}

/**
 * Arrête la vérification d'alerte.
 */
function stopAlertPolling() {
  if (alertInterval) {
    clearInterval(alertInterval);
    alertInterval = null;
  }
}


if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", () => {
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    if (isLight) {
      document.documentElement.removeAttribute("data-theme");
      localStorage.setItem("theme", "dark");
      themeIcon.classList.remove("fa-sun");
      themeIcon.classList.add("fa-moon");
    } else {
      document.documentElement.setAttribute("data-theme", "light");
      localStorage.setItem("theme", "light");
      themeIcon.classList.remove("fa-moon");
      themeIcon.classList.add("fa-sun");
    }
  });
}

fetchModelInfo();
initTheme();

