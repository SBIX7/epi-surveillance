"""
app/vision.py — Moteur de détection EPI (SmartSafety Vision)

Modèle par défaut : keremberke/yolov8n-ppe-detection (HuggingFace)
  Classes : Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
            Person, Safety Cone, Safety Vest, machinery, vehicle

Fallback  : yolov8n.pt (COCO 80 classes, aucune classe EPI)
  → Dans ce cas, un avertissement est affiché dans le flux vidéo.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ── Modèle PPE spécialisé (HuggingFace) ──────────────────────────────────
PPE_MODEL_HF   = "keremberke/yolov8n-ppe-detection"
FALLBACK_MODEL = "yolov8n.pt"

# ── Synonymes de la classe "personne" dans les modèles YOLOv8 ─────────────
# Couvre : COCO, modèles PPE Roboflow, datasets chantier personnalisés.
PERSON_LABELS = [
    "person", "people", "worker", "human", "man", "woman",
    "ouvrier", "personne", "employee", "staff", "operator",
    "pedestrian", "individual",
]


class ModelLoadError(Exception):
    """Levée quand le modèle YOLO ne peut pas être chargé."""


class SafetyGearEngine:
    """
    Moteur principal de détection EPI.

    Logique d'association EPI ↔ Personne :
    ─────────────────────────────────────
    On calcule le ratio d'intersection entre la bounding-box de la personne
    et celle de l'EPI :

        overlap = Aire(Intersection) / Aire(Personne)

    Si ce ratio > GEAR_OVERLAP_THRESHOLD, l'EPI est considéré porté.
    """

    # ── Groupes d'EPI obligatoires ────────────────────────────────────────
    REQUIRED_GEAR_GROUPS: Dict[str, List[str]] = {
        "Casque": [
            "helmet", "hardhat", "hard-hat", "headgear",
            # classes modele keremberke PPE
            "Hardhat",
        ],
        "Gilet": [
            "vest", "jacket", "safety vest", "safety-vest",
            "hi-vis", "hivis",
            # classes modele keremberke PPE
            "Safety Vest",
        ],
        "Gants": [
            "glove", "gloves", "gant", "gants",
        ],
        "Lunettes": [
            "goggles", "goggle", "glasses", "eye protection",
            "lunettes",
        ],
        "Masque": [
            "mask", "masque", "face mask", "respirator",
            # classes modele keremberke PPE
            "Mask",
        ],
        "Chaussures": [
            "shoe", "shoes", "boot", "boots", "safety shoe",
            "chaussure", "chaussures",
        ],
    }

    # ── Détections NÉGATIVES (personne SANS EPI) ─────────────────────────
    # Certains modèles PPE fournissent aussi des classes de non-conformité.
    # Si l'on détecte une de ces classes SUR une personne, on force le manquant.
    NEGATIVE_GEAR_MAP: Dict[str, str] = {
        # Casque
        "no-hardhat":     "Casque",
        "no hardhat":     "Casque",
        "no helmet":      "Casque",
        "no-helmet":      "Casque",
        "no_helmet":      "Casque",
        # Gilet
        "no-safety vest": "Gilet",
        "no safety vest": "Gilet",
        "no-vest":        "Gilet",
        "no vest":        "Gilet",
        "no_vest":        "Gilet",
        # Gants
        "no-glove":       "Gants",
        "no glove":       "Gants",
        "no_glove":       "Gants",
        "no-gloves":      "Gants",
        "no gloves":      "Gants",
        "no_gloves":      "Gants",
        # Lunettes
        "no-goggles":     "Lunettes",
        "no goggles":     "Lunettes",
        "no_goggles":     "Lunettes",
        "no-glasses":     "Lunettes",
        "no glasses":     "Lunettes",
        "no_glasses":     "Lunettes",
        # Masque
        "no-mask":        "Masque",
        "no mask":        "Masque",
        "no_mask":        "Masque",
        # Chaussures
        "no-shoe":        "Chaussures",
        "no shoe":        "Chaussures",
        "no_shoe":        "Chaussures",
        "no-shoes":       "Chaussures",
        "no shoes":       "Chaussures",
        "no_shoes":       "Chaussures",
    }

    # ── Couleurs BGR ─────────────────────────────────────────────────────
    COLOR_OK      = (0, 200, 80)    # Vert  — EPI complet
    COLOR_MISSING = (30, 30, 220)   # Rouge — EPI manquant
    COLOR_GEAR    = (20, 150, 255)  # Orange — bounding box EPI
    COLOR_WARN    = (0, 200, 200)   # Cyan  — info générale

    # ── Seuil d'intersection ─────────────────────────────────────────────
    GEAR_OVERLAP_THRESHOLD: float = 0.02

    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, model_path: Optional[str] = None):
        self.model_path: str = model_path or PPE_MODEL_HF
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self._epi_classes_available: bool = False
        self._person_class_ids: set = set()  # initialisé avant load

        self._try_load_with_fallback()

    # ── Chargement avec fallback ──────────────────────────────────────────
    def _try_load_with_fallback(self) -> None:
        """
        Tente de charger le modèle PPE spécialisé depuis HuggingFace.
        Si indisponible (hors-ligne, etc.), bascule sur yolov8n.pt COCO.
        """
        for path in (self.model_path, FALLBACK_MODEL):
            try:
                print(f"[SmartSafety] Chargement du modele : {path}")
                self.load_model(path)
                if self._epi_classes_available:
                    print(f"[SmartSafety] Modele EPI operationnel ({len(self.class_names)} classes)")
                    return
                else:
                    print(f"[SmartSafety] ATTENTION : '{path}' ne contient pas de classes EPI.")
                    if path == self.model_path and path != FALLBACK_MODEL:
                        print(f"[SmartSafety] Tentative avec le modele PPE HuggingFace...")
                        # Si le modèle demandé n'a pas de classes EPI,
                        # on tente quand même le PPE HuggingFace
                        try:
                            self.load_model(PPE_MODEL_HF)
                            if self._epi_classes_available:
                                return
                        except Exception:
                            pass
                    return  # Dernier recours : on garde ce modèle
            except ModelLoadError as exc:
                print(f"[SmartSafety] Impossible de charger '{path}' : {exc}")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Charge (ou recharge) un modèle YOLOv8.
        Met à jour `_epi_classes_available` et `_person_class_ids`.

        Raises: ModelLoadError
        """
        self.model_path = model_path or self.model_path
        if not self.model_path:
            raise ModelLoadError("Aucun chemin de modele fourni.")
        try:
            self.model = YOLO(str(self.model_path))
            self.class_names = {int(k): str(v) for k, v in self.model.names.items()}
            self._epi_classes_available = self._check_epi_classes()
            self._person_class_ids: set = self._detect_person_class_ids()

            # ── Log de toutes les classes pour diagnostic ────────────────
            all_cls = list(self.class_names.values())
            print(f"[SmartSafety] Modele charge : {len(all_cls)} classes")
            print(f"[SmartSafety] Classes : {all_cls}")
            print(f"[SmartSafety] Classes personne detectees : {[self.class_names[i] for i in self._person_class_ids]}")
            print(f"[SmartSafety] Classes EPI disponibles   : {self._epi_classes_available}")
        except Exception as exc:
            raise ModelLoadError(f"Impossible de charger le modele YOLO : {exc}") from exc

    # ── Validation des classes EPI ────────────────────────────────────────
    def _check_epi_classes(self) -> bool:
        """
        Retourne True si le modèle possède au moins une classe EPI connue.
        Un modèle COCO standard (yolov8n.pt) retournera False.
        """
        return any(
            self._is_positive_gear(name) or self._is_negative_gear(name)
            for name in self.class_names.values()
        )

    def _detect_person_class_ids(self) -> set:
        """
        Identifie automatiquement les IDs de classes correspondant à une
        "personne" dans le modèle chargé, en cherchant parmi PERSON_LABELS.
        Ainsi, si le modèle utilise 'worker', 'human', 'ouvrier', etc.,
        ils seront tous traités comme des personnes.
        """
        ids = set()
        for cls_id, name in self.class_names.items():
            if any(kw in name.lower() for kw in PERSON_LABELS):
                ids.add(cls_id)
        if not ids:
            print("[SmartSafety] ATTENTION : aucune classe 'personne' trouvee dans ce modele !")
            print(f"[SmartSafety] Classes disponibles : {list(self.class_names.values())}")
        return ids

    def get_model_info(self) -> dict:
        """Retourne un dict résumant les classes du modèle actuel (pour l'API)."""
        person_cls = [self.class_names[i] for i in self._person_class_ids]
        gear_cls   = [n for n in self.class_names.values() if self._is_positive_gear(n) or self._is_negative_gear(n)]
        other_cls  = [n for n in self.class_names.values() if n not in person_cls and n not in gear_cls]
        return {
            "model_path":           self.model_path,
            "total_classes":        len(self.class_names),
            "all_classes":          list(self.class_names.values()),
            "person_classes":       person_cls,
            "epi_classes":          gear_cls,
            "other_classes":        other_cls,
            "epi_classes_available": self._epi_classes_available,
            "person_detected":      len(person_cls) > 0,
        }

    def _is_positive_gear(self, label: str) -> bool:
        """Vérifie si l'étiquette correspond à un EPI POSITIF (porté)."""
        text = label.lower()
        return any(
            kw.lower() in text
            for keywords in self.REQUIRED_GEAR_GROUPS.values()
            for kw in keywords
        )

    def _is_negative_gear(self, label: str) -> bool:
        """Vérifie si l'étiquette correspond à une ABSENCE d'EPI."""
        text = label.lower()
        return any(neg in text for neg in self.NEGATIVE_GEAR_MAP)

    def _gear_groups_from_label(self, label: str) -> List[str]:
        """Retourne les groupes EPI correspondant à une étiquette positive."""
        text = label.lower()
        return [
            group
            for group, keywords in self.REQUIRED_GEAR_GROUPS.items()
            if any(kw.lower() in text for kw in keywords)
        ]

    def _missing_from_negative_label(self, label: str) -> Optional[str]:
        """
        Si l'étiquette signale l'ABSENCE d'un EPI (ex: 'NO-Hardhat'),
        retourne le groupe manquant ('Casque'). Sinon None.
        """
        text = label.lower()
        for neg, group in self.NEGATIVE_GEAR_MAP.items():
            if neg in text:
                return group
        return None

    # ── Algorithme d'intersection ─────────────────────────────────────────
    @staticmethod
    def _intersection_ratio(
        person_box: Tuple[int, int, int, int],
        gear_box:   Tuple[int, int, int, int],
    ) -> float:
        """
        Ratio : Aire(Intersection) / Aire(Personne).
        Mesure la proportion de la bbox personne couverte par l'EPI.
        """
        ix1 = max(person_box[0], gear_box[0])
        iy1 = max(person_box[1], gear_box[1])
        ix2 = min(person_box[2], gear_box[2])
        iy2 = min(person_box[3], gear_box[3])

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        person_area = max(
            1,
            (person_box[2] - person_box[0]) * (person_box[3] - person_box[1]),
        )
        return inter_area / person_area

    # ── Dessin texte avec fond ────────────────────────────────────────────
    @staticmethod
    def _draw_label_bg(
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        color: Tuple[int, int, int],
        font_scale: float = 0.52,
        thickness: int = 1,
    ) -> None:
        """Affiche un texte avec un rectangle de fond coloré."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = origin
        pad = 4
        cv2.rectangle(
            frame,
            (x - pad, y - th - base - pad),
            (x + tw + pad, y + base),
            color,
            cv2.FILLED,
        )
        cv2.putText(frame, text, (x, y - base), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # ── Overlay avertissement (modele non-EPI) ────────────────────────────
    def _draw_no_epi_model_warning(self, frame: np.ndarray) -> None:
        """
        Affiche un bandeau d'avertissement si le modèle chargé
        ne possède aucune classe EPI (ex : yolov8n.pt COCO).
        """
        h, w = frame.shape[:2]

        # Fond rouge semi-transparent en haut
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 180), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame,
            "MODELE COCO : aucune classe EPI detectee !",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "-> Chargez un modele PPE specialise (.pt) via la sidebar",
            (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (200, 200, 255), 1, cv2.LINE_AA,
        )

    # ── Inférence + annotation ────────────────────────────────────────────
    def predict_and_annotate(self, frame: np.ndarray, conf: float = 0.25) -> np.ndarray:
        """
        Réalise l'inférence YOLOv8 et annote la frame :
          • VERT   — personne avec tous ses EPI / EPI positif détecté seul
          • ROUGE  — personne avec EPI manquant(s) / EPI négatif détecté seul
          • ORANGE — bounding box EPI détecté

        Si le modèle n'a pas de classes EPI, affiche un bandeau d'avertissement.
        Si le modèle n'a pas de classe personne (modèle EPI-only), chaque
        détection EPI est annotée indépendamment (positif = vert, négatif = rouge).
        """
        if self.model is None:
            raise ModelLoadError("Aucun modele YOLO charge.")

        try:
            results = self.model(frame, conf=conf, imgsz=640, verbose=False)[0]
        except Exception as exc:
            raise RuntimeError(f"Erreur inference YOLO : {exc}") from exc

        persons: List[Dict] = []
        pos_gears: List[Dict] = []   # EPI positifs (porté)
        neg_gears: List[Dict] = []   # EPI négatifs (absent détecté)

        # ── Tri des détections ───────────────────────────────────────────
        for box in results.boxes:
            cls   = int(box.cls[0])
            score = float(box.conf[0])
            label = self.class_names.get(cls, f"cls_{cls}")
            xyxy  = box.xyxy[0].cpu().numpy().astype(int)
            bbox  = tuple(xyxy.tolist())

            # Détection personne : flexible — compare l'ID de classe
            if cls in self._person_class_ids:
                persons.append({"box": bbox, "score": score, "label": label})
            elif self._is_positive_gear(label):
                pos_gears.append({"box": bbox, "score": score, "label": label})
            elif self._is_negative_gear(label):
                neg_gears.append({"box": bbox, "score": score, "label": label})

        annotation = frame.copy()
        no_person_model = len(self._person_class_ids) == 0

        # ── Avertissement si modèle COCO ─────────────────────────────────
        if not self._epi_classes_available:
            self._draw_no_epi_model_warning(annotation)

        # ══════════════════════════════════════════════════════════════════
        # MODE A : modèle EPI-only (pas de classe personne)
        # Chaque détection EPI est affichée directement avec son statut.
        # ══════════════════════════════════════════════════════════════════
        if no_person_model and self._epi_classes_available:
            # EPI positifs → vert (conformité OK)
            for gear in pos_gears:
                x1, y1, x2, y2 = gear["box"]
                cv2.rectangle(annotation, (x1, y1), (x2, y2), self.COLOR_OK, 2)
                self._draw_label_bg(
                    annotation,
                    f"✓ {gear['label']} {gear['score']:.0%}",
                    (x1, max(22, y1 - 4)),
                    self.COLOR_OK,
                )

            # EPI négatifs → rouge (non-conformité)
            for gear in neg_gears:
                x1, y1, x2, y2 = gear["box"]
                cv2.rectangle(annotation, (x1, y1), (x2, y2), self.COLOR_MISSING, 2)
                self._draw_label_bg(
                    annotation,
                    f"✗ {gear['label']} {gear['score']:.0%}",
                    (x1, max(22, y1 - 4)),
                    self.COLOR_MISSING,
                )

            if not pos_gears and not neg_gears:
                cv2.putText(
                    annotation,
                    "Aucun EPI detecte",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    self.COLOR_WARN, 2, cv2.LINE_AA,
                )
            return annotation

        # ══════════════════════════════════════════════════════════════════
        # MODE B : modèle avec classe personne (comportement standard)
        # ══════════════════════════════════════════════════════════════════

        # ── Dessiner les EPI positifs (orange) ──────────────────────────
        for gear in pos_gears:
            x1, y1, x2, y2 = gear["box"]
            cv2.rectangle(annotation, (x1, y1), (x2, y2), self.COLOR_GEAR, 2)
            self._draw_label_bg(
                annotation,
                f"{gear['label']} {gear['score']:.0%}",
                (x1, max(22, y1 - 4)),
                self.COLOR_GEAR,
            )

        # ── Dessiner les EPI négatifs (rouge clair) ──────────────────────
        for gear in neg_gears:
            x1, y1, x2, y2 = gear["box"]
            cv2.rectangle(annotation, (x1, y1), (x2, y2), (0, 80, 200), 2)
            self._draw_label_bg(
                annotation,
                f"{gear['label']} {gear['score']:.0%}",
                (x1, max(22, y1 - 4)),
                (0, 80, 200),
            )

        # ── Évaluation par personne ──────────────────────────────────────
        for person in persons:
            pbox = person["box"]
            found_groups: set = set()
            forced_missing: set = set()

            # 1. Associations via EPI POSITIFS
            for gear in pos_gears:
                ratio = self._intersection_ratio(pbox, gear["box"])
                if ratio > self.GEAR_OVERLAP_THRESHOLD:
                    found_groups.update(self._gear_groups_from_label(gear["label"]))

            # 2. Associations via EPI NÉGATIFS (force un manquant)
            for gear in neg_gears:
                ratio = self._intersection_ratio(pbox, gear["box"])
                if ratio > self.GEAR_OVERLAP_THRESHOLD:
                    missing_group = self._missing_from_negative_label(gear["label"])
                    if missing_group:
                        forced_missing.add(missing_group)

            # Un groupe détecté négativement n'est pas considéré OK
            found_groups -= forced_missing

            # Filtrer les groupes requis à ceux disponibles dans ce modèle
            available_groups = set()
            for g in self.REQUIRED_GEAR_GROUPS:
                if any(
                    self._is_positive_gear(n) or self._is_negative_gear(n)
                    for n in self.class_names.values()
                    if g.lower() in self._gear_groups_from_label(n)
                       or any(kw.lower() in n.lower() for kw in self.REQUIRED_GEAR_GROUPS.get(g, []))
                       or any(g.lower() in (self.NEGATIVE_GEAR_MAP.get(neg, "")).lower()
                              for neg in self.NEGATIVE_GEAR_MAP
                              if neg in n.lower())
                ):
                    available_groups.add(g)

            required = available_groups if available_groups else set(self.REQUIRED_GEAR_GROUPS.keys())
            missing  = list((required - found_groups) | forced_missing)

            if missing:
                color      = self.COLOR_MISSING
                label_text = "MANQUANT: " + ", ".join(missing)
            else:
                color      = self.COLOR_OK
                label_text = "EPI OK"

            x1, y1, x2, y2 = pbox
            cv2.rectangle(annotation, (x1, y1), (x2, y2), color, 3)
            self._draw_label_bg(
                annotation,
                label_text,
                (x1, max(22, y1 - 4)),
                color,
                font_scale=0.58,
            )

        # ── Message global si personne non détectée ──────────────────────
        if not persons:
            cv2.putText(
                annotation,
                "Aucune personne detectee",
                (12, 34 + (90 if not self._epi_classes_available else 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                self.COLOR_WARN, 2, cv2.LINE_AA,
            )

        return annotation
