# -*- coding: utf-8 -*-
"""
GPX Profiler — Endurance Core Lab
Outil d'analyse d'un fichier GPX : profil altimétrique, gradients,
statistiques par tranche, visualisation filtrée, analyse de tronçon,
détection de surface OSM et carte interactive.

Auteur  : alanb
Version : 2.1
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================

import math                          # Fonctions mathématiques (radians, sqrt, asin…)

import folium                        # Carte interactive (rendu HTML côté client)
import gpxpy                         # Lecture et parsing de fichiers GPX
import numpy as np                   # Calcul vectoriel sur tableaux
import pandas as pd                  # Manipulation de tableaux de données (DataFrame)
import plotly.graph_objects as go    # Construction de graphiques interactifs
import requests                      # Requêtes HTTP vers l'API Overpass (OSM)
import streamlit as st               # Framework d'interface web interactive
import streamlit.components.v1 as components  # Injection de JS pour la synchronisation curseur
from streamlit_folium import st_folium  # Intégration Folium (carte statique, onglet séparé)


# =============================================================================
# SECTION 2 — CONSTANTES GLOBALES
# =============================================================================

# Bornes des tranches de gradient en pourcentage (%).
# np.inf représente « plus de 30 % ».
GRADIENT_BINS = [0, 5, 10, 15, 20, 25, 30, np.inf]

# Libellés lisibles associés à chaque tranche (affichés dans le tableau et la légende).
BIN_LABELS = ["0-5 %", "5-10 %", "10-15 %", "15-20 %", "20-25 %", "25-30 %", "> 30 %"]

# Palette de couleurs pour les portions en MONTÉE (tons chauds : jaune clair → rouge brun).
ASCENT_COLORS = [
    "#FFF8CC",  # très clair (quasi blanc chaud)
    "#FFE082",  # jaune clair
    "#FFCA5F",  # jaune chaud
    "#F6A03A",  # orange clair
    "#E0701F",  # orange soutenu
    "#C04A0F",  # rouge-orange
    "#AA3D00",  # référence (rouge brun profond)
]

# Palette de couleurs pour les portions en DESCENTE (tons froids : vert clair → vert forêt).
DESCENT_COLORS = [
    "#F4F8F1",  # très léger vert (quasi blanc)
    "#DCE8CF",  # vert clair
    "#C3D9AE",  # vert doux
    "#A8C686",  # référence
    "#7FA35C",  # vert moyen
    "#5F7F44",  # vert soutenu
    "#3F5A2C",  # vert foncé / forest
]

# Types de surface reconnus et leurs couleurs RGBA associées.
SURFACE_TYPES = ["Route", "Gravel", "Technique", "Inconnu"]
SURFACE_COLORS = {
    "Route"     : "rgba(80,80,80,0.7)",     # Gris foncé
    "Gravel"    : "rgba(194,154,108,0.7)",  # Beige/sable
    "Technique" : "rgba(139,90,43,0.7)",    # Brun terre
    "Inconnu"   : "rgba(180,180,180,0.4)",  # Gris clair
}

# Style commun des axes Plotly (couleur et taille de police).
AXIS_STYLE = dict(
    showgrid=False,
    zeroline=False,
    tickfont=dict(size=14, color="#453E3B"),
    title_font=dict(size=16, color="#453E3B"),
)

# Couleur principale du thème graphique.
THEME_COLOR = "#453E3B"


# =============================================================================
# SECTION 3 — FONCTIONS UTILITAIRES (calculs géographiques et formatage)
# =============================================================================

def format_hhmm(hours):
    """
    Convertit un temps en heures (float) en chaîne lisible HH:MM.

    Paramètres d'entrée :
        hours (float) : Durée en heures.

    Paramètres de sortie :
        (str) : Format "XhMM" (ex. : "2h35").
    """
    total_min = int(round(hours * 60))
    return f"{total_min // 60}h{total_min % 60:02d}"


def haversine(lat1, lon1, lat2, lon2):
    """
    Calcule la distance en mètres entre deux points GPS sur la surface terrestre.

    Utilise la formule de Haversine qui tient compte de la courbure de la Terre.
    Adaptée pour des distances de l'ordre de quelques kilomètres à quelques
    centaines de kilomètres (erreur négligeable pour les courses de trail).

    Paramètres d'entrée :
        lat1 (float) : Latitude du point de départ, en degrés décimaux.
        lon1 (float) : Longitude du point de départ, en degrés décimaux.
        lat2 (float) : Latitude du point d'arrivée, en degrés décimaux.
        lon2 (float) : Longitude du point d'arrivée, en degrés décimaux.

    Paramètres de sortie :
        (float) : Distance en mètres entre les deux points GPS.
    """
    R = 6_371_000  # Rayon moyen de la Terre en mètres.

    # Conversion des coordonnées de degrés en radians.
    phi1    = math.radians(lat1)
    phi2    = math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Terme intermédiaire de Haversine : carré du demi-angle sous-tendu par l'arc.
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    # Distance angulaire en radians, puis conversion en mètres.
    return 2 * R * math.asin(math.sqrt(a))


# =============================================================================
# SECTION 4 — FONCTIONS DE PARSING ET TRAITEMENT DES DONNÉES GPX
# =============================================================================

@st.cache_data
def parse_gpx_file(file):
    """
    Lit un fichier GPX et extrait la liste ordonnée des points GPS (lat, lon, alt).

    Tente d'abord de lire les tracks (traces enregistrées, format le plus courant),
    puis les routes (waypoints reliés manuellement) si aucun track n'est trouvé.

    Paramètres d'entrée :
        file : Objet fichier ouvert en lecture (compatible avec gpxpy.parse),
               typiquement le résultat de st.file_uploader().

    Paramètres de sortie :
        points (list of tuples) : Liste de tuples (latitude, longitude, elevation).
                                  Les coordonnées sont en degrés décimaux, l'altitude en mètres.
    """
    gpx    = gpxpy.parse(file)
    points = []

    # Parcours des tracks (format standard des montres GPS).
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append((pt.latitude, pt.longitude, pt.elevation))

    # Repli sur les routes si aucun track n'est présent.
    if not points:
        for route in gpx.routes:
            for pt in route.points:
                points.append((pt.latitude, pt.longitude, pt.elevation))

    return points


@st.cache_data
def compute_cumulative_distances_and_elevations(points):
    """
    Calcule les distances cumulées et les altitudes pour chaque point GPS.

    Parcourt les points dans l'ordre et accumule la distance en utilisant
    la formule de Haversine entre chaque paire de points consécutifs.
    Les altitudes manquantes (None) sont remplacées par la dernière valeur connue.

    Paramètres d'entrée :
        points (list of tuples) : Liste de tuples (latitude, longitude, elevation)
                                  telle que retournée par parse_gpx_file().

    Paramètres de sortie :
        distances  (np.ndarray, float64) : Distances cumulées depuis le départ, en mètres.
                                           distances[0] = 0, distances[-1] = distance totale.
        elevations (np.ndarray, float64) : Altitudes correspondantes, en mètres.
                                           Même longueur que distances.
    """
    # Initialisation avec le premier point (distance = 0).
    distances  = [0.0]
    elevations = [points[0][2] if points[0][2] is not None else 0.0]

    # Calcul incrémental : distance entre chaque paire de points consécutifs.
    for i in range(1, len(points)):
        dist_step = haversine(
            points[i - 1][0], points[i - 1][1],
            points[i][0],     points[i][1]
        )
        distances.append(distances[-1] + dist_step)

        # Altitude du point courant (si None, on réutilise la valeur précédente).
        elev = points[i][2] if points[i][2] is not None else elevations[-1]
        elevations.append(elev)

    return np.array(distances), np.array(elevations)


@st.cache_data
def discretize_into_100m_segments(distances, elevations):
    """
    Rééchantillonne le profil altimétrique tous les 100 mètres et calcule
    le gradient (pente en %) de chaque tronçon de 100 m.

    Les points GPS d'un fichier GPX sont irrégulièrement espacés. Cette fonction
    interpole le profil à intervalles réguliers de 100 m pour homogénéiser
    l'analyse des pentes.

    Paramètres d'entrée :
        distances  (np.ndarray) : Distances cumulées en mètres.
        elevations (np.ndarray) : Altitudes correspondantes en mètres.

    Paramètres de sortie :
        breakpoints  (np.ndarray) : Points de discrétisation en mètres (0, 100, 200, …).
        elev_interp  (np.ndarray) : Altitudes interpolées aux breakpoints, en mètres.
        seg_start    (np.ndarray) : Distance de début de chaque segment (mètres).
        seg_end      (np.ndarray) : Distance de fin de chaque segment (mètres).
        seg_mid      (np.ndarray) : Distance du milieu de chaque segment (mètres).
        seg_dist     (np.ndarray) : Longueur de chaque segment (mètres, ≈100 m sauf le dernier).
        seg_gradient (np.ndarray) : Gradient de chaque segment en % (positif=montée, négatif=descente).
    """
    total_distance = distances[-1]

    # Points de discrétisation espacés de 100 m, avec le dernier point du parcours inclus.
    breakpoints = np.arange(0, total_distance, 100)
    if breakpoints[-1] < total_distance:
        breakpoints = np.append(breakpoints, total_distance)

    # Interpolation linéaire de l'altitude aux breakpoints.
    elev_interp = np.interp(breakpoints, distances, elevations)

    # Longueur et dénivelé de chaque segment entre deux breakpoints consécutifs.
    seg_dist = np.diff(breakpoints)
    seg_elev = np.diff(elev_interp)

    # Gradient en pourcentage : (dénivelé / distance horizontale) × 100.
    seg_gradient = (seg_elev / seg_dist) * 100

    seg_start = breakpoints[:-1]
    seg_end   = breakpoints[1:]
    seg_mid   = (seg_start + seg_end) / 2

    return breakpoints, elev_interp, seg_start, seg_end, seg_mid, seg_dist, seg_gradient


@st.cache_data
def compute_global_stats(distances, elev_bp):
    """
    Calcule les statistiques globales du parcours : distance totale, D+ et D-.

    Paramètres d'entrée :
        distances (np.ndarray) : Distances cumulées en mètres.
        elev_bp   (np.ndarray) : Altitudes interpolées aux breakpoints de 100 m.

    Paramètres de sortie :
        total_dist_km (float) : Distance totale du parcours en kilomètres.
        total_pos     (float) : Dénivelé positif total (somme des montées) en mètres.
        total_neg     (float) : Dénivelé négatif total (somme des descentes, valeur positive) en mètres.
    """
    total_dist_km = distances[-1] / 1000

    # Variations d'altitude entre chaque paire de breakpoints consécutifs.
    elev_diffs = np.diff(elev_bp)

    total_pos = sum(max(0.0, d) for d in elev_diffs)
    total_neg = sum(abs(min(0.0, d)) for d in elev_diffs)

    return total_dist_km, total_pos, total_neg


# =============================================================================
# SECTION 5 — FONCTIONS DE CLASSIFICATION OSM ET DÉTECTION DE SURFACE
# =============================================================================

@st.cache_data
def classify_osm_tags(highway, surface, tracktype, sac_scale):
    """
    Convertit les tags OSM d'un chemin en type de surface parmi les 4 catégories.

    Paramètres d'entrée :
        highway   (str) : Tag OSM « highway ».
        surface   (str) : Tag OSM « surface ».
        tracktype (str) : Tag OSM « tracktype ».
        sac_scale (str) : Tag OSM « sac_scale ».

    Paramètres de sortie :
        (str) : Type de surface parmi « Route », « Gravel », « Technique », « Inconnu ».
    """
    # Route : surface dure ou infrastructure routière classifiée.
    if surface in ("asphalt", "paved", "concrete", "tar"):
        return "Route"
    if highway in ("primary", "secondary", "tertiary", "residential",
                   "unclassified", "road", "service", "trunk", "motorway"):
        return "Route"

    # Gravel : piste roulante non goudronnée.
    if surface in ("gravel", "fine_gravel", "compacted", "pebblestone", "crushed_limestone"):
        return "Gravel"
    if tracktype in ("grade1", "grade2"):
        return "Gravel"
    if highway == "cycleway":
        return "Gravel"

    # Technique : sentier ou terrain difficile.
    if surface in ("dirt", "ground", "grass", "mud", "rock", "sand", "earth", "woodchips"):
        return "Technique"
    if tracktype in ("grade3", "grade4", "grade5"):
        return "Technique"
    if highway in ("path", "footway", "bridleway", "steps"):
        return "Technique"
    if sac_scale in ("mountain_hiking", "demanding_mountain_hiking",
                     "alpine_hiking", "demanding_alpine_hiking",
                     "difficult_alpine_hiking"):
        return "Technique"

    # Track sans information précise → Gravel par défaut (hypothèse conservative).
    if highway == "track":
        return "Gravel"

    return "Inconnu"


def fetch_surface_from_osm(points, seg_start, seg_dist):
    """
    Récupère le type de surface le long du parcours via un UNIQUE appel
    à l'API Overpass (OpenStreetMap).

    Télécharge tous les chemins dans la bounding box du parcours, puis affecte
    à chaque segment de 100 m le chemin OSM le plus proche.

    Paramètres d'entrée :
        points    (list of tuples) : Points GPS bruts (lat, lon, elev).
        seg_start (np.ndarray)     : Distances de début de chaque segment en mètres.
        seg_dist  (np.ndarray)     : Longueur de chaque segment en mètres.

    Paramètres de sortie :
        seg_surface (list of str) : Type de surface pour chaque segment.
    """
    # ── Calcul de la bounding box du parcours (avec marge ~500 m) ───────────
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    lat_min = min(lats) - 0.005
    lat_max = max(lats) + 0.005
    lon_min = min(lons) - 0.005
    lon_max = max(lons) + 0.005

    query = f"""
    [out:json][timeout:60];
    way({lat_min},{lon_min},{lat_max},{lon_max})
        ["highway"];
    out body geom;
    """

    # Liste de serveurs Overpass de secours (on essaie dans l'ordre).
    OVERPASS_SERVERS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    elements = []
    for server_url in OVERPASS_SERVERS:
        try:
            resp = requests.get(
                server_url,
                params={"data": query},
                headers={"Accept": "application/json"},
                timeout=60
            )
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            break  # Succès : on sort de la boucle de repli.
        except Exception:
            continue  # On essaie le serveur suivant.

    if not elements:
        st.warning("Aucun chemin OSM trouvé ou tous les serveurs sont inaccessibles.")
        return ["Inconnu"] * len(seg_start)

    # ── Extraction des segments OSM avec leurs tags et géométrie ─────────────
    # Pour chaque way OSM, on crée un point représentatif au milieu de chaque
    # arête (paire de nœuds consécutifs) avec son type de surface associé.
    osm_segments = []

    for el in elements:
        tags      = el.get("tags", {})
        highway   = tags.get("highway",   "")
        surface   = tags.get("surface",   "")
        tracktype = tags.get("tracktype", "")
        sac_scale = tags.get("sac_scale", "")
        surf_type = classify_osm_tags(highway, surface, tracktype, sac_scale)

        geometry = el.get("geometry", [])
        if len(geometry) < 2:
            continue

        for k in range(len(geometry) - 1):
            lat_mid = (geometry[k]["lat"] + geometry[k + 1]["lat"]) / 2
            lon_mid = (geometry[k]["lon"] + geometry[k + 1]["lon"]) / 2
            osm_segments.append((lat_mid, lon_mid, surf_type))

    if not osm_segments:
        return ["Inconnu"] * len(seg_start)

    # ── Conversion en arrays NumPy pour le calcul vectorisé ─────────────────
    osm_lats  = np.array([s[0] for s in osm_segments])
    osm_lons  = np.array([s[1] for s in osm_segments])
    osm_types = [s[2] for s in osm_segments]

    # ── Distances cumulées GPS pour interpolation lat/lon ───────────────────
    dist_brut = [0.0]
    for i in range(1, len(points)):
        dist_brut.append(dist_brut[-1] + haversine(
            points[i - 1][0], points[i - 1][1],
            points[i][0],     points[i][1]
        ))
    dist_brut = np.array(dist_brut)
    gps_lats  = np.array([p[0] for p in points])
    gps_lons  = np.array([p[1] for p in points])

    # ── Affectation : pour chaque segment GPX de 100 m, trouver le way OSM──
    # le plus proche (distance euclidienne approx. en degrés, suffisante à
    # cette échelle pour identifier le bon chemin).
    seg_surface = []

    for i in range(len(seg_start)):
        # Coordonnées géographiques du milieu du segment GPX.
        dist_mid = float(seg_start[i]) + float(seg_dist[i]) / 2
        lat_seg  = float(np.interp(dist_mid, dist_brut, gps_lats))
        lon_seg  = float(np.interp(dist_mid, dist_brut, gps_lons))

        # Distance approx. (en degrés²) à tous les segments OSM — pas besoin
        # de racine carrée pour trouver le minimum.
        dlat  = osm_lats - lat_seg
        dlon  = osm_lons - lon_seg
        dist2 = dlat ** 2 + dlon ** 2

        # Seuil de tolérance : ~200 m ≈ 0.002 degré.
        idx_min = int(np.argmin(dist2))
        if dist2[idx_min] < 0.002 ** 2:
            seg_surface.append(osm_types[idx_min])
        else:
            seg_surface.append("Inconnu")

    return seg_surface


def build_surface_table(seg_start, seg_dist, seg_surface):
    """
    Construit le tableau récapitulatif des tronçons par type de surface.

    Les segments consécutifs de même surface sont fusionnés en un seul tronçon.

    Paramètres d'entrée :
        seg_start   (np.ndarray)  : Distances de début de chaque segment en mètres.
        seg_dist    (np.ndarray)  : Longueur de chaque segment en mètres.
        seg_surface (list of str) : Type de surface de chaque segment.

    Paramètres de sortie :
        df (pd.DataFrame) : Tableau avec colonnes km_début, km_fin, distance, surface.
    """
    rows          = []
    current_surf  = seg_surface[0]
    current_start = seg_start[0]
    current_dist  = 0.0

    for i in range(len(seg_surface)):
        if seg_surface[i] == current_surf:
            current_dist += seg_dist[i]
        else:
            # Fermeture du tronçon précédent et ouverture d'un nouveau.
            rows.append({
                "Début (km)"      : round(current_start / 1000, 2),
                "Fin (km)"        : round((current_start + current_dist) / 1000, 2),
                "Distance (km)"   : round(current_dist / 1000, 2),
                "Type de surface" : current_surf,
            })
            current_surf  = seg_surface[i]
            current_start = seg_start[i]
            current_dist  = seg_dist[i]

    # Fermeture du dernier tronçon.
    rows.append({
        "Début (km)"      : round(current_start / 1000, 2),
        "Fin (km)"        : round((current_start + current_dist) / 1000, 2),
        "Distance (km)"   : round(current_dist / 1000, 2),
        "Type de surface" : current_surf,
    })

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 6 — FONCTIONS DE COLORIMÉTRIE ET DE CONSTRUCTION DES SHAPES PLOTLY
# =============================================================================

def get_bin_index(gradient_value):
    """
    Retourne l'indice de la tranche (bin) correspondant à une valeur de gradient.

    Paramètres d'entrée :
        gradient_value (float) : Valeur du gradient en %, positif ou négatif.

    Paramètres de sortie :
        (int) : Indice entre 0 et len(BIN_LABELS)-1 indiquant la tranche de gradient.
    """
    for i, upper_bound in enumerate(GRADIENT_BINS[1:]):
        if abs(gradient_value) <= upper_bound:
            return i
    # Cas de dépassement (ne devrait pas se produire avec np.inf en dernière borne).
    return len(GRADIENT_BINS) - 2


def get_segment_color(gradient_value, alpha=1.0):
    """
    Retourne la couleur RGBA associée à un gradient donné.

    Choisit la palette chaude (montée) ou froide (descente) selon le signe du gradient,
    puis sélectionne la teinte correspondant à la tranche de gradient.

    Paramètres d'entrée :
        gradient_value (float) : Gradient du segment en % (positif = montée, négatif = descente).
        alpha          (float) : Opacité entre 0.0 (transparent) et 1.0 (opaque). Défaut : 1.0.

    Paramètres de sortie :
        (str) : Couleur au format CSS « rgba(r, g, b, alpha) ».
    """
    idx       = get_bin_index(gradient_value)
    hex_color = ASCENT_COLORS[idx] if gradient_value >= 0 else DESCENT_COLORS[idx]

    # Décomposition hexadécimale en composantes RGB.
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    return f"rgba({r},{g},{b},{alpha})"


def build_gradient_shapes(seg_start, seg_end, seg_gradient, distances, elevations, alpha=0.75):
    """
    Construit la liste des rectangles colorés (shapes Plotly) représentant
    les segments de 100 m colorés selon leur gradient.

    Chaque segment est représenté par un rectangle vertical translucide dont la
    hauteur correspond à l'altitude maximale du segment et la largeur à l'étendue
    horizontale (~100 m = 0,1 km).

    Paramètres d'entrée :
        seg_start    (np.ndarray) : Distances de début de chaque segment, en mètres.
        seg_end      (np.ndarray) : Distances de fin de chaque segment, en mètres.
        seg_gradient (np.ndarray) : Gradient de chaque segment en %.
        distances    (np.ndarray) : Distances cumulées des points GPS d'origine, en mètres.
        elevations   (np.ndarray) : Altitudes des points GPS d'origine, en mètres.
        alpha        (float)      : Opacité des rectangles. Défaut : 0.75.

    Paramètres de sortie :
        shapes (list of dict) : Liste de shapes Plotly (type « rect »), prêts pour
                                fig.update_layout(shapes=shapes).
    """
    shapes = []

    for i in range(len(seg_gradient)):
        grad  = seg_gradient[i]
        color = get_segment_color(grad, alpha=alpha)

        # Altitude interpolée en début et fin de segment.
        elev_start = np.interp(seg_start[i], distances, elevations)
        elev_end   = np.interp(seg_end[i],   distances, elevations)

        # La hauteur du rectangle va jusqu'au point le plus haut du segment.
        y_top = max(elev_start, elev_end)

        shapes.append(dict(
            type="rect",
            x0=seg_start[i] / 1000,  # Début en km (axe X).
            x1=seg_end[i]   / 1000,  # Fin en km.
            y0=0,                     # Base (altitude zéro).
            y1=y_top,                 # Sommet (altitude max du segment).
            fillcolor=color,
            line=dict(width=0),       # Pas de bordure visible.
            layer="above",
        ))

    return shapes


def build_filtered_shapes(seg_start, seg_end, seg_gradient, distances, elevations,
                           threshold_up, threshold_down):
    """
    Construit les rectangles colorés pour la visualisation filtrée (onglet 2).

    Seuls les segments dépassant les seuils définis par l'utilisateur sont colorés :
        - Orange/rouge : montées avec gradient >= threshold_up.
        - Vert         : descentes avec |gradient| >= threshold_down.

    Paramètres d'entrée :
        seg_start      (np.ndarray) : Distances de début de chaque segment en mètres.
        seg_end        (np.ndarray) : Distances de fin de chaque segment en mètres.
        seg_gradient   (np.ndarray) : Gradient de chaque segment en %.
        distances      (np.ndarray) : Distances cumulées GPS en mètres.
        elevations     (np.ndarray) : Altitudes GPS en mètres.
        threshold_up   (int)        : Seuil minimal de gradient pour colorer les montées (%).
        threshold_down (int)        : Seuil minimal (valeur abs.) pour colorer les descentes (%).

    Paramètres de sortie :
        shapes (list of dict) : Liste de shapes Plotly filtrés selon les seuils.
    """
    shapes = []

    for i in range(len(seg_gradient)):
        grad = seg_gradient[i]

        elev_start = np.interp(seg_start[i], distances, elevations)
        elev_end   = np.interp(seg_end[i],   distances, elevations)
        y_top      = max(elev_start, elev_end)

        if grad >= threshold_up:
            # Montée dépassant le seuil → rectangle orange/rouge translucide.
            color = "rgba(170, 61, 0, 0.7)"
        elif grad <= -threshold_down:
            # Descente dépassant le seuil → rectangle vert translucide.
            color = "rgba(168, 198, 134, 0.7)"
        else:
            continue  # Segment ne dépassant aucun seuil : on ne l'ajoute pas.

        shapes.append(dict(
            type="rect",
            x0=seg_start[i] / 1000,
            x1=seg_end[i]   / 1000,
            y0=0,
            y1=y_top,
            fillcolor=color,
            line=dict(width=0),
            layer="above",
        ))

    return shapes


# =============================================================================
# SECTION 7 — FONCTIONS DE MISE EN FORME DES GRAPHIQUES PLOTLY
# =============================================================================

def apply_axes_style(fig):
    """
    Applique le style visuel commun (axes, grille, bordures) à une figure Plotly.

    Factorise les appels répétitifs update_xaxes / update_yaxes / add_vline /
    add_hline qui sont identiques sur tous les graphiques de l'application.

    Paramètres d'entrée :
        fig (go.Figure) : Figure Plotly à styler (modifiée en place).

    Paramètres de sortie :
        fig (go.Figure) : La même figure, avec le style appliqué.
    """
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)

    # Bordures sur les axes (ligne verticale à x=0, ligne horizontale à y=0).
    fig.add_vline(x=0, line_color=THEME_COLOR, line_width=1.0, layer="above")
    fig.add_hline(y=0, line_color=THEME_COLOR, line_width=1.0, layer="above")

    return fig


@st.cache_data
def build_elevation_figure(distances, elevations, seg_start, seg_end, seg_gradient):
    """
    Construit le graphique Plotly du profil altimétrique complet avec
    les rectangles colorés par tranche de gradient.

    Superpose :
        1. Une courbe de profil altimétrique en fond.
        2. Des rectangles colorés pour chaque segment de 100 m.
        3. Des entrées de légende pour chaque tranche de gradient (montée & descente).

    Paramètres d'entrée :
        distances    (np.ndarray) : Distances cumulées des points GPS en mètres.
        elevations   (np.ndarray) : Altitudes des points GPS en mètres.
        seg_start    (np.ndarray) : Distances de début de chaque segment en mètres.
        seg_end      (np.ndarray) : Distances de fin de chaque segment en mètres.
        seg_gradient (np.ndarray) : Gradient de chaque segment en %.

    Paramètres de sortie :
        fig (go.Figure) : Figure Plotly complète, prête pour st.plotly_chart().
    """
    fig = go.Figure()

    # Courbe de base : profil altimétrique.
    fig.add_trace(go.Scatter(
        x=distances / 1000,
        y=elevations,
        mode="lines",
        line=dict(color=THEME_COLOR, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(84,110,122,0.0)",
        name="Altitude",
        hovertemplate="Distance : %{x:.2f} km<br>Altitude : %{y:.0f} m<extra></extra>",
        showlegend=False
    ))

    # Rectangles colorés par tranche de gradient.
    shapes = build_gradient_shapes(
        seg_start, seg_end, seg_gradient,
        distances, elevations,
        alpha=0.75
    )

    # Entrées de légende (traces invisibles servant uniquement à la légende).
    for i, (label, ascent_color, descent_color) in enumerate(
        zip(BIN_LABELS, ASCENT_COLORS, DESCENT_COLORS)
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=ascent_color, symbol="square"),
            name=f"montée {label}",
            legendgroup=f"asc_{i}",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=descent_color, symbol="square"),
            name=f"descente {label}",
            legendgroup=f"desc_{i}",
            showlegend=True
        ))

    fig.update_layout(
        shapes=shapes,
        xaxis_title="Distance (km)",
        yaxis_title="Altitude (m)",
        legend=dict(
            title="Gradient",
            orientation="v",
            x=1.01, y=1,
            font=dict(size=13, color=THEME_COLOR),
            title_font=dict(size=14, color=THEME_COLOR)
        ),
        height=540,
        margin=dict(l=60, r=200, t=30, b=60),
        hovermode="x",
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="solid",
            spikecolor=THEME_COLOR,
            hoverformat=".2f",   # 2 chiffres après la virgule
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13, color=THEME_COLOR)
    )

    apply_axes_style(fig)

    return fig


@st.cache_data
def build_filtered_figure(distances, elevations, seg_start, seg_end, seg_gradient,
                           threshold_up, threshold_down):
    """
    Construit le graphique Plotly de la visualisation filtrée par seuil de gradient.

    Superpose le profil altimétrique de base avec des zones colorées uniquement
    pour les segments dont la pente dépasse les seuils choisis par l'utilisateur.

    Paramètres d'entrée :
        distances      (np.ndarray) : Distances cumulées GPS en mètres.
        elevations     (np.ndarray) : Altitudes GPS en mètres.
        seg_start      (np.ndarray) : Distances de début de chaque segment en mètres.
        seg_end        (np.ndarray) : Distances de fin de chaque segment en mètres.
        seg_gradient   (np.ndarray) : Gradient de chaque segment en %.
        threshold_up   (int)        : Seuil minimal de gradient pour les montées (%).
        threshold_down (int)        : Seuil minimal (valeur abs.) pour les descentes (%).

    Paramètres de sortie :
        fig (go.Figure) : Figure Plotly prête pour st.plotly_chart().
    """
    fig = go.Figure()

    # Profil altimétrique de base.
    fig.add_trace(go.Scatter(
        x=distances / 1000,
        y=elevations,
        mode="lines",
        line=dict(color=THEME_COLOR, width=2),
        fill="tozeroy",
        fillcolor="rgba(69, 62, 59, 0.10)",
        name="Altitude",
        hovertemplate="Distance : %{x:.2f} km<br>Altitude : %{y:.0f} m<extra></extra>"
    ))

    # Rectangles filtrés selon les seuils.
    shapes = build_filtered_shapes(
        seg_start, seg_end, seg_gradient,
        distances, elevations,
        threshold_up, threshold_down
    )

    # Entrées de légende explicitant les seuils actifs.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=14, color="rgba(170, 61, 0, 0.7)", symbol="square"),
        name=f"Montée >= {threshold_up} %"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=14, color="rgba(168, 198, 134, 0.7)", symbol="square"),
        name=f"Descente >= {threshold_down} %"
    ))

    fig.update_layout(
        shapes=shapes,
        xaxis_title="Distance (km)",
        yaxis_title="Altitude (m)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=60, r=60, t=30, b=60),
        hovermode="x",
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="solid",
            spikecolor=THEME_COLOR,
            hoverformat=".2f",   # 2 chiffres après la virgule
        ),
        plot_bgcolor="white"
    )

    apply_axes_style(fig)

    return fig


def build_troncon_figure(distances, elevations, troncon_start, troncon_end):
    """
    Construit le graphique de profil pour un tronçon délimité par l'utilisateur.

    Affiche le profil complet en fond (grisé) et met en évidence le tronçon
    sélectionné avec deux lignes verticales de délimitation.

    Paramètres d'entrée :
        distances     (np.ndarray) : Distances cumulées GPS en mètres.
        elevations    (np.ndarray) : Altitudes GPS en mètres.
        troncon_start (float)      : Début du tronçon en km.
        troncon_end   (float)      : Fin du tronçon en km.

    Paramètres de sortie :
        fig (go.Figure) : Figure Plotly prête pour st.plotly_chart().
    """
    fig = go.Figure()

    # Profil complet en fond (grisé, trait fin).
    fig.add_trace(go.Scatter(
        x=distances / 1000,
        y=elevations,
        mode="lines",
        line=dict(color=THEME_COLOR, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(69, 62, 59, 0.15)",
        name="Profil complet",
        hovertemplate="Distance : %{x:.2f} km<br>Altitude : %{y:.0f} m<extra></extra>"
    ))

    # Mise en évidence du tronçon sélectionné (zone colorée, trait plus épais).
    mask_troncon = (distances / 1000 >= troncon_start) & (distances / 1000 <= troncon_end)
    fig.add_trace(go.Scatter(
        x=distances[mask_troncon] / 1000,
        y=elevations[mask_troncon],
        mode="lines",
        line=dict(color=THEME_COLOR, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(69, 62, 59, 0.3)",
        name="Tronçon sélectionné",
        hoverinfo="skip",
    ))

    # Lignes verticales de délimitation du tronçon.
    fig.add_vline(
        x=troncon_start,
        line=dict(color="#A8C686", width=2, dash="dash"),
        annotation_text=f"Début {troncon_start:.1f} km",
        annotation_position="top"
    )
    fig.add_vline(
        x=troncon_end,
        line=dict(color="#AA3D00", width=2, dash="dash"),
        annotation_text=f"Fin {troncon_end:.1f} km",
        annotation_position="top"
    )

    fig.update_layout(
        xaxis_title="Distance (km)",
        yaxis_title="Altitude (m)",
        height=450,
        margin=dict(l=60, r=60, t=30, b=60),
        hovermode="x",
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="solid",
            spikecolor=THEME_COLOR,
            hoverformat=".2f",   # 2 chiffres après la virgule
        ),
        plot_bgcolor="white"
    )

    apply_axes_style(fig)

    return fig


def build_surface_figure(distances, elevations, seg_start, seg_end, seg_surface):
    """
    Construit le profil altimétrique coloré par type de surface.

    Paramètres d'entrée :
        distances   (np.ndarray)  : Distances cumulées GPS en mètres.
        elevations  (np.ndarray)  : Altitudes GPS en mètres.
        seg_start   (np.ndarray)  : Distances de début de chaque segment en mètres.
        seg_end     (np.ndarray)  : Distances de fin de chaque segment en mètres.
        seg_surface (list of str) : Type de surface de chaque segment.

    Paramètres de sortie :
        fig (go.Figure) : Figure Plotly prête à afficher.
    """
    fig = go.Figure()

    # Courbe de base.
    fig.add_trace(go.Scatter(
        x=distances / 1000,
        y=elevations,
        mode="lines",
        line=dict(color=THEME_COLOR, width=2),
        fill="tozeroy",
        fillcolor="rgba(69,62,59,0.08)",
        name="Altitude",
        showlegend=False,
        hovertemplate="Distance : %{x:.2f} km<br>Altitude : %{y:.0f} m<extra></extra>"
    ))

    # Rectangles colorés par type de surface.
    shapes = []
    for i in range(len(seg_surface)):
        color  = SURFACE_COLORS[seg_surface[i]]
        elev_s = float(np.interp(seg_start[i], distances, elevations))
        elev_e = float(np.interp(seg_end[i],   distances, elevations))
        y_top  = max(elev_s, elev_e)
        shapes.append(dict(
            type="rect",
            x0=seg_start[i] / 1000,
            x1=seg_end[i]   / 1000,
            y0=0, y1=y_top,
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        ))

    # Entrées de légende (traces invisibles pour chaque type de surface).
    for surf_type, color_rgba in SURFACE_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color_rgba, symbol="square"),
            name=surf_type
        ))

    fig.update_layout(
        shapes=shapes,
        xaxis_title="Distance (km)",
        yaxis_title="Altitude (m)",
        height=480,
        margin=dict(l=60, r=60, t=30, b=60),
        hovermode="x",
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="solid",
            spikecolor=THEME_COLOR,
            hoverformat=".2f",   # 2 chiffres après la virgule
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.08, font=dict(size=14, color=THEME_COLOR))
    )

    apply_axes_style(fig)

    return fig


# =============================================================================
# SECTION 8 — FONCTIONS DE CALCUL STATISTIQUE
# =============================================================================

def compute_gradient_statistics(seg_gradient, seg_dist, total_pos, total_neg):
    """
    Calcule les statistiques détaillées par tranche de gradient.

    Pour chaque tranche définie dans GRADIENT_BINS / BIN_LABELS, calcule :
        - Distance totale des segments en montée et en descente.
        - Pourcentage de cette distance par rapport au total monté/descendu.
        - Dénivelé cumulé (D+ et D-) sur ces segments.
        - Pourcentage du D+/D- total du parcours.

    Paramètres d'entrée :
        seg_gradient (np.ndarray) : Gradient de chaque segment en %.
        seg_dist     (np.ndarray) : Longueur de chaque segment en mètres.
        total_pos    (float)      : Dénivelé positif total du parcours en mètres.
        total_neg    (float)      : Dénivelé négatif total du parcours en mètres.

    Paramètres de sortie :
        rows (list of dict) : Une ligne par tranche de gradient, avec les statistiques
                              formatées (str) pour affichage dans un DataFrame.
    """
    total_dist_up_m   = seg_dist[seg_gradient > 0].sum()
    total_dist_down_m = seg_dist[seg_gradient < 0].sum()

    rows = []

    for i, label in enumerate(BIN_LABELS):
        lo = GRADIENT_BINS[i]
        hi = GRADIENT_BINS[i + 1]

        # Masques booléens pour les segments de montée et de descente dans la tranche.
        if hi == np.inf:
            mask_up   = seg_gradient >= lo
            mask_down = seg_gradient <= -lo
        else:
            mask_up   = (seg_gradient >= lo)  & (seg_gradient < hi)
            mask_down = (seg_gradient <= -lo) & (seg_gradient > -hi)

        # Distances en km pour cette tranche.
        dist_up_km   = seg_dist[mask_up].sum()   / 1000
        dist_down_km = seg_dist[mask_down].sum() / 1000

        # Dénivelés : gradient (%) × distance (m) / 100 → mètres de dénivelé.
        elev_up   = (seg_gradient[mask_up]   * seg_dist[mask_up]   / 100).sum()
        elev_down = abs((seg_gradient[mask_down] * seg_dist[mask_down] / 100).sum())

        # Pourcentages de distance.
        pct_up   = (dist_up_km   / (total_dist_up_m   / 1000) * 100) if total_dist_up_m   > 0 else 0
        pct_down = (dist_down_km / (total_dist_down_m / 1000) * 100) if total_dist_down_m > 0 else 0

        # Pourcentages de dénivelé.
        pct_dplus  = (elev_up   / total_pos * 100) if total_pos > 0 else 0
        pct_dminus = (elev_down / total_neg * 100) if total_neg > 0 else 0

        rows.append({
            "Gradient"             : label,
            "Dist. montée (km)"    : f"{dist_up_km:.2f}",
            "% montée (dist)"      : f"{pct_up:.1f} %",
            "D+ (m)"               : f"{elev_up:.0f}",
            "% D+ total"           : f"{pct_dplus:.1f} %",
            "Dist. descente (km)"  : f"{dist_down_km:.2f}",
            "% descente (dist)"    : f"{pct_down:.1f} %",
            "D- (m)"               : f"{elev_down:.0f}",
            "% D- total"           : f"{pct_dminus:.1f} %",
        })

    return rows


def build_stats_dataframe(rows):
    """
    Construit le DataFrame pandas final pour le tableau statistique,
    en ajoutant une ligne TOTAL en bas.

    Paramètres d'entrée :
        rows (list of dict) : Lignes statistiques issues de compute_gradient_statistics().

    Paramètres de sortie :
        df_display (pd.DataFrame) : DataFrame avec les lignes de données + ligne TOTAL.
    """
    df = pd.DataFrame(rows)

    # Somme des colonnes numériques (les valeurs sont stockées en str, on les convertit).
    total_dist_up   = sum(float(r["Dist. montée (km)"])   for r in rows)
    total_dist_down = sum(float(r["Dist. descente (km)"]) for r in rows)
    total_dplus     = sum(float(r["D+ (m)"])              for r in rows)
    total_dminus    = sum(float(r["D- (m)"])              for r in rows)

    total_row = {
        "Gradient"             : "TOTAL",
        "Dist. montée (km)"    : f"{total_dist_up:.2f}",
        "% montée (dist)"      : "100 %",
        "D+ (m)"               : f"{total_dplus:.0f}",
        "% D+ total"           : "100 %",
        "Dist. descente (km)"  : f"{total_dist_down:.2f}",
        "% descente (dist)"    : "100 %",
        "D- (m)"               : f"{total_dminus:.0f}",
        "% D- total"           : "100 %",
    }

    df_display = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df_display


def style_stats_row(row, df_columns):
    """
    Applique un style CSS à chaque cellule d'une ligne du tableau de statistiques.

    Les colonnes de montée reçoivent un fond chaud (selon la tranche),
    les colonnes de descente un fond froid, et la ligne TOTAL un style spécial en gras.

    Destinée à être passée à df.style.apply(axis=1).

    Paramètres d'entrée :
        row        (pd.Series) : Une ligne du DataFrame.
        df_columns (Index)     : Colonnes du DataFrame.

    Paramètres de sortie :
        styles (list of str) : Liste de chaînes CSS, une par colonne.
    """
    # Ligne TOTAL : mise en évidence spéciale.
    if row["Gradient"] == "TOTAL":
        return [
            "font-weight: bold; background-color: #ECEFF1; border-top: 2px solid #607D8B"
        ] * len(df_columns)

    # Lignes normales : couleurs de la tranche correspondante.
    idx           = BIN_LABELS.index(row["Gradient"])
    ascent_color  = ASCENT_COLORS[idx]
    descent_color = DESCENT_COLORS[idx]

    # Colonnes de montée en fond chaud, colonnes de descente en fond froid.
    ASCENT_COLS  = {"Dist. montée (km)", "% montée (dist)", "D+ (m)", "% D+ total"}
    DESCENT_COLS = {"Dist. descente (km)", "% descente (dist)", "D- (m)", "% D- total"}

    styles = []
    for col in df_columns:
        if col in ASCENT_COLS:
            styles.append(f"background-color: {ascent_color}22; color: #222")
        elif col in DESCENT_COLS:
            styles.append(f"background-color: {descent_color}22; color: #222")
        else:
            styles.append("")
    return styles


def compute_altitude_stats(seg_start, seg_dist, elev_bp, total_dist_km):
    """
    Calcule la distance et la proportion du parcours passée au-dessus
    de différents seuils d'altitude.

    Paramètres d'entrée :
        seg_start     (np.ndarray) : Distances de début de chaque segment en mètres.
        seg_dist      (np.ndarray) : Longueur de chaque segment en mètres.
        elev_bp       (np.ndarray) : Altitudes aux breakpoints en mètres.
        total_dist_km (float)      : Distance totale du parcours en km.

    Paramètres de sortie :
        stats (list of dict) : Une ligne par seuil d'altitude avec distance (km)
                               et proportion (%).
    """
    SEUILS = [1500, 2000, 2500, 3000]  # Seuils d'altitude en mètres.

    stats = []
    for seuil in SEUILS:
        dist_au_dessus = 0.0

        for i in range(len(seg_dist)):
            # Altitude moyenne du segment (moyenne des deux breakpoints encadrants).
            alt_moy = (elev_bp[i] + elev_bp[i + 1]) / 2.0
            if alt_moy >= seuil:
                dist_au_dessus += seg_dist[i]

        dist_km    = dist_au_dessus / 1000
        proportion = (dist_km / total_dist_km * 100) if total_dist_km > 0 else 0.0

        stats.append({
            "Seuil d'altitude" : f"Supérieur à {seuil} m",
            "Distance (km)"    : round(dist_km, 2),
            "Proportion (%)"   : round(proportion, 1),
        })

    return stats


def compute_troncon_stats(seg_gradient, seg_dist, seg_mid, distances, elevations,
                           start_m, end_m, troncon_dist_km):
    """
    Calcule les statistiques (D+, D-, gradient moyen) d'un tronçon délimité.

    Paramètres d'entrée :
        seg_gradient    (np.ndarray) : Gradient de chaque segment en %.
        seg_dist        (np.ndarray) : Longueur de chaque segment en mètres.
        seg_mid         (np.ndarray) : Distance du milieu de chaque segment en mètres.
        distances       (np.ndarray) : Distances cumulées GPS en mètres.
        elevations      (np.ndarray) : Altitudes GPS en mètres.
        start_m         (float)      : Début du tronçon en mètres.
        end_m           (float)      : Fin du tronçon en mètres.
        troncon_dist_km (float)      : Distance totale du tronçon en km.

    Paramètres de sortie :
        troncon_dplus          (float) : D+ sur le tronçon en mètres.
        troncon_dminus         (float) : D- sur le tronçon en mètres.
        troncon_gradient_moyen (float) : Gradient moyen net du tronçon en %.
    """
    mask_troncon = (seg_mid >= start_m) & (seg_mid <= end_m)

    # Altitudes interpolées aux bornes exactes du tronçon.
    elev_at_start = float(np.interp(start_m, distances, elevations))
    elev_at_end   = float(np.interp(end_m,   distances, elevations))

    # D+ et D- segment par segment.
    elev_diffs_troncon = seg_gradient[mask_troncon] * seg_dist[mask_troncon] / 100
    troncon_dplus      = sum(max(0.0, d) for d in elev_diffs_troncon)
    troncon_dminus     = sum(abs(min(0.0, d)) for d in elev_diffs_troncon)

    # Gradient moyen = dénivelé net / distance horizontale totale du tronçon.
    denivele_net           = elev_at_end - elev_at_start
    troncon_gradient_moyen = (
        (denivele_net / (troncon_dist_km * 1000)) * 100
        if troncon_dist_km > 0 else 0
    )

    return troncon_dplus, troncon_dminus, troncon_gradient_moyen


def compute_filtered_stats(seg_gradient, seg_dist, total_dist_km, total_pos, total_neg,
                            threshold_up, threshold_down):
    """
    Calcule les statistiques des segments filtrés par les seuils de gradient.

    Paramètres d'entrée :
        seg_gradient   (np.ndarray) : Gradient de chaque segment en %.
        seg_dist       (np.ndarray) : Longueur de chaque segment en mètres.
        total_dist_km  (float)      : Distance totale du parcours en km.
        total_pos      (float)      : D+ total du parcours en mètres.
        total_neg      (float)      : D- total du parcours en mètres.
        threshold_up   (int)        : Seuil de gradient pour les montées (%).
        threshold_down (int)        : Seuil (valeur abs.) pour les descentes (%).

    Paramètres de sortie :
        d_up_km         (float) : Distance cumulée des montées filtrées, en km.
        d_down_km       (float) : Distance cumulée des descentes filtrées, en km.
        dplus_filtered  (float) : D+ cumulé des montées filtrées, en mètres.
        dminus_filtered (float) : D- cumulé des descentes filtrées, en mètres.
    """
    mask_up   = seg_gradient >= threshold_up
    mask_down = seg_gradient <= -threshold_down

    d_up_km   = seg_dist[mask_up].sum()   / 1000
    d_down_km = seg_dist[mask_down].sum() / 1000

    dplus_filtered  = (seg_gradient[mask_up]   * seg_dist[mask_up]   / 100).sum()
    dminus_filtered = abs((seg_gradient[mask_down] * seg_dist[mask_down] / 100).sum())

    return d_up_km, d_down_km, dplus_filtered, dminus_filtered


# =============================================================================
# SECTION 9 — CARTE INTERACTIVE
# =============================================================================

@st.cache_data
def build_route_map(points):
    """
    Construit la carte Folium du tracé GPX.

    Optimisations pour la fluidité :
        - Tuile légère OpenStreetMap (standard) utilisée par défaut, à la place
          d'OpenTopoMap qui sollicite un serveur tiers plus lent.
        - La carte est mise en cache (@st.cache_data) : elle n'est reconstruite
          que lorsque les points changent (i.e. nouveau fichier GPX).
        - Le rendu du fond de carte et du tracé est effectué côté client (HTML/JS).
          Seul le premier chargement fait un round-trip serveur ; les zooms et
          déplacements sont entièrement gérés par Leaflet dans le navigateur.

    Note : pour supprimer les callbacks inutiles vers le serveur Streamlit lors
    des interactions utilisateur, st_folium est appelé avec returned_objects=[]
    dans render_map_section().

    Paramètres d'entrée :
        points (list of tuples) : Points GPS bruts (lat, lon, elev).

    Paramètres de sortie :
        m (folium.Map) : Carte Folium prête pour st_folium().
    """
    lat_center = np.mean([p[0] for p in points])
    lon_center = np.mean([p[1] for p in points])

    m = folium.Map(
        location=[lat_center, lon_center],
        tiles="OpenStreetMap",   # Tuile légère (serveur rapide, pas de latence OSM topo).
        zoom_start=12,
        prefer_canvas=True       # Rendu Canvas plus performant que SVG pour les tracés denses.
    )

    # Tracé du parcours GPX.
    folium.PolyLine(
        locations=[(p[0], p[1]) for p in points],
        color="#AA3D00",
        weight=3.0
    ).add_to(m)

    # Marqueur de départ (vert) et d'arrivée (rouge).
    folium.CircleMarker(
        location=(points[0][0],  points[0][1]),
        radius=6, color="#3F5A2C", fill=True, fill_color="#A8C686",
        tooltip="Départ"
    ).add_to(m)
    folium.CircleMarker(
        location=(points[-1][0], points[-1][1]),
        radius=6, color="#AA3D00", fill=True, fill_color="#E0701F",
        tooltip="Arrivée"
    ).add_to(m)

    return m


# =============================================================================
# SECTION 9bis — CARTE ET PROFIL SYNCHRONISÉS (Leaflet + Plotly liés en JS)
# =============================================================================
#
# Contrairement à la V2.0 (carte Plotly Scattermapbox intégrée au même subplot
# que le profil), cette version utilise une carte Leaflet "brute" (la même
# technologie que Folium, mais instanciée directement en JavaScript) et un
# graphique Plotly de profil séparé, tous deux injectés dans UN SEUL composant
# HTML Streamlit. Comme les deux objets vivent dans le même document JS, ils
# peuvent communiquer directement : zoom/pan à la molette restent natifs sur
# la carte (Leaflet gère ça nativement, indépendamment du profil), et la carte
# peut occuper toute la hauteur souhaitée sans contrainte de subplot.
#
# Schéma de communication :
#   profil (Plotly) --plotly_hover--> JS --marker.setLatLng()--> carte (Leaflet)
#

def build_synced_profile_figure(distances, elevations):
    """
    Construit la figure Plotly du profil altimétrique seul, destinée à être
    affichée sous la carte Leaflet dans le composant HTML synchronisé.

    Cette figure est volontairement simple (pas de rectangles de gradient)
    pour rester légère et garantir une bonne réactivité du hover, qui pilote
    en temps réel le marqueur sur la carte.

    Paramètres d'entrée :
        distances  (np.ndarray) : Distances cumulées GPS en mètres.
        elevations (np.ndarray) : Altitudes GPS en mètres.

    Paramètres de sortie :
        fig (go.Figure) : Figure Plotly du profil, prête pour fig.to_html().
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=distances / 1000,
        y=elevations,
        mode="lines",
        line=dict(color=THEME_COLOR, width=2),
        fill="tozeroy",
        fillcolor="rgba(69, 62, 59, 0.12)",
        name="Altitude",
        hovertemplate="Distance : %{x:.2f} km<br>Altitude : %{y:.0f} m<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Distance (km)",
        yaxis_title="Altitude (m)",
        height=400,
        margin=dict(l=60, r=62, t=20, b=50),
        hovermode="x",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    apply_axes_style(fig)

    return fig


def render_synced_map_section(points, distances, elevations):
    """
    Affiche la carte Leaflet (zoom molette natif, pleine hauteur) et le profil
    altimétrique synchronisés dans un unique composant HTML.

    Détail d'implémentation :
        - La carte est créée directement en JavaScript via L.map(...) et
          L.tileLayer(...), exactement comme le ferait Folium en interne,
          mais sans passer par st_folium : on garde donc le zoom/pan natif
          de Leaflet, totalement indépendant du rendu du profil.
        - Le profil est exporté en HTML autonome via fig.to_html() et injecté
          juste en dessous dans le même document.
        - Un marqueur Leaflet (L.circleMarker) est créé une fois, caché par
          défaut, et déplacé par JS à chaque événement "plotly_hover" émis
          par le profil — aucun callback Streamlit n'est déclenché, donc
          aucune latence serveur pendant l'interaction.

    Paramètres d'entrée :
        points     (list of tuples) : Points GPS bruts (lat, lon, elev).
        distances  (np.ndarray)     : Distances cumulées GPS en mètres.
        elevations (np.ndarray)     : Altitudes GPS en mètres.
    """
    
    st.caption(
        "Déplacez la souris sur le profil altimétrique : la position "
        "correspondante s'affiche sur la carte en temps réel. "
    )

    # Hauteur de la carte réglable, pour répondre au besoin "carte plus grande".
    map_height_px = 500

    # Données géographiques sérialisées en JSON pour le JS.
    lats_json = [round(p[0], 6) for p in points]
    lons_json = [round(p[1], 6) for p in points]
    lat_center = float(np.mean(lats_json))
    lon_center = float(np.mean(lons_json))

    # Figure du profil, exportée en HTML autonome (sans tag <html>/<body>,
    # uniquement la div + le script d'initialisation Plotly).
    fig_profile = build_synced_profile_figure(distances, elevations)
    profile_html = fig_profile.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="synced-profile-div"
    )
    
    # ── Construction du composant HTML complet (carte Leaflet + profil) ────
    # Note : Leaflet est chargé depuis son CDN officiel, au même titre que
    # Plotly. Les deux librairies coexistent sans conflit dans ce document.
    full_component_html = f"""
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <div style="max-width: 100%; margin: 0 auto;">
        <div id="synced-leaflet-map" style="width:93%; height:{map_height_px}px; margin: 0 auto;"></div>
        {profile_html}
    </div>

    <script>
        // ── Initialisation de la carte Leaflet (zoom/pan natifs) ────────
        var map = L.map("synced-leaflet-map").setView([{lat_center}, {lon_center}], 13);

        // Tuile OpenStreetMap standard : rapide, sans clé API requise.
        L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
            maxZoom: 19,
            attribution: "&copy; OpenStreetMap contributors"
        }}).addTo(map);

        // Tracé du parcours GPX.
        var lats = {lats_json};
        var lons = {lons_json};
        var routeLatLngs = lats.map(function(lat, i) {{ return [lat, lons[i]]; }});

        var routeLine = L.polyline(routeLatLngs, {{
            color: "#AA3D00",
            weight: 3.5
        }}).addTo(map);

        // Ajustement automatique du zoom pour cadrer tout le parcours.
        map.fitBounds(routeLine.getBounds(), {{ padding: [20, 20] }});

        // Marqueurs de départ et d'arrivée.
        L.circleMarker(routeLatLngs[0], {{
            radius: 7, color: "#3F5A2C", fillColor: "#A8C686", fillOpacity: 1
        }}).addTo(map).bindTooltip("Départ");

        L.circleMarker(routeLatLngs[routeLatLngs.length - 1], {{
            radius: 7, color: "#AA3D00", fillColor: "#E0701F", fillOpacity: 1
        }}).addTo(map).bindTooltip("Arrivée");

        // Marqueur de position, déplacé dynamiquement au survol du profil.
        // Caché initialement (opacity 0) jusqu'au premier survol.
        var cursorMarker = L.circleMarker(routeLatLngs[0], {{
            radius: 7,
            color: "#453e3b",
            weight: 2,
            fillColor: "#453e3b",
            fillOpacity: 0.95,
            opacity: 0
        }}).addTo(map);

        // ── Liaison du profil Plotly vers le marqueur Leaflet ───────────
        var profileDiv = document.getElementById("synced-profile-div");

        profileDiv.on("plotly_hover", function(eventData) {{
            var pt = eventData.points[0];
            if (!pt) return;

            var idx = pt.pointIndex;
            if (idx < 0 || idx >= lats.length) return;

            var newLatLng = [lats[idx], lons[idx]];
            cursorMarker.setLatLng(newLatLng);
            cursorMarker.setStyle({{ opacity: 1 }});
        }});

        profileDiv.on("plotly_unhover", function(eventData) {{
            // Le marqueur reste affiché à sa dernière position connue,
            // pour une lecture plus confortable après le survol.
        }});
    </script>
    """

    components.html(
        full_component_html,
        height=map_height_px + 420,   # Marge pour la figure du profil sous la carte.
        scrolling=False
    )


# =============================================================================
# SECTION 10 — RENDUS DES ONGLETS (extraction depuis main)
# =============================================================================

def render_tab1(distances, elevations, seg_start, seg_end, seg_dist,
                seg_gradient, elev_bp, total_pos, total_neg, total_dist_km):
    """
    Affiche le contenu de l'onglet 1 : profil altimétrique + statistiques.

    Paramètres d'entrée :
        distances     (np.ndarray) : Distances cumulées GPS en mètres.
        elevations    (np.ndarray) : Altitudes GPS en mètres.
        seg_start     (np.ndarray) : Distances de début de chaque segment en mètres.
        seg_end       (np.ndarray) : Distances de fin de chaque segment en mètres.
        seg_dist      (np.ndarray) : Longueur de chaque segment en mètres.
        seg_gradient  (np.ndarray) : Gradient de chaque segment en %.
        elev_bp       (np.ndarray) : Altitudes aux breakpoints en mètres.
        total_pos     (float)      : D+ total du parcours en mètres.
        total_neg     (float)      : D- total du parcours en mètres.
        total_dist_km (float)      : Distance totale en km.
    """
    # Graphique principal : profil + rectangles colorés par gradient.
    fig_profile = build_elevation_figure(
        distances, elevations,
        seg_start, seg_end, seg_gradient
    )
    st.plotly_chart(fig_profile, use_container_width=True)

    # Tableau statistique par tranche de gradient.
    stat_rows = compute_gradient_statistics(seg_gradient, seg_dist, total_pos, total_neg)
    df_stats  = build_stats_dataframe(stat_rows)

    st.dataframe(
        df_stats.style.apply(
            lambda row: style_stats_row(row, df_stats.columns),
            axis=1
        ),
        use_container_width=True,
        hide_index=True
    )

    # Métriques d'altitude par seuil (1500 m, 2000 m, 2500 m, 3000 m).
    st.markdown("#### Distance par tranche d'altitude")
    altitude_stats = compute_altitude_stats(seg_start, seg_dist, elev_bp, total_dist_km)

    col_alt = st.columns(4)
    for i, row in enumerate(altitude_stats):
        col_alt[i].metric(
            label=row["Seuil d'altitude"],
            value=f"{row['Distance (km)']} km",
            delta=f"{row['Proportion (%)']} % du parcours"
        )


def render_tab2(distances, elevations, seg_start, seg_end, seg_dist, seg_gradient,
                total_dist_km, total_pos, total_neg):
    """
    Affiche le contenu de l'onglet 2 : visualisation filtrée par gradient.

    Paramètres d'entrée :
        distances     (np.ndarray) : Distances cumulées GPS en mètres.
        elevations    (np.ndarray) : Altitudes GPS en mètres.
        seg_start     (np.ndarray) : Distances de début de chaque segment en mètres.
        seg_end       (np.ndarray) : Distances de fin de chaque segment en mètres.
        seg_dist      (np.ndarray) : Longueur de chaque segment en mètres.
        seg_gradient  (np.ndarray) : Gradient de chaque segment en %.
        total_dist_km (float)      : Distance totale en km.
        total_pos     (float)      : D+ total en mètres.
        total_neg     (float)      : D- total en mètres.
    """
    # Sliders de seuil de gradient (montée et descente).
    col_slider1, col_slider2 = st.columns(2)

    with col_slider1:
        threshold_up = st.slider(
            "Afficher les montées avec gradient >=",
            min_value=0, max_value=35, value=10, step=1,
            help="Seules les portions montantes dont le gradient est >= cette valeur seront colorées."
        )

    with col_slider2:
        threshold_down = st.slider(
            "Afficher les descentes avec gradient <= -",
            min_value=0, max_value=35, value=10, step=1,
            help="Seules les descentes dont le gradient (valeur abs.) est >= cette valeur seront colorées."
        )

    # Graphique filtré.
    fig_filtered = build_filtered_figure(
        distances, elevations,
        seg_start, seg_end, seg_gradient,
        threshold_up, threshold_down
    )
    st.plotly_chart(fig_filtered, use_container_width=True)

    # Métriques résumées (distances et dénivelés filtrés).
    d_up_km, d_down_km, dplus_filtered, dminus_filtered = compute_filtered_stats(
        seg_gradient, seg_dist,
        total_dist_km, total_pos, total_neg,
        threshold_up, threshold_down
    )

    col_a, col_b = st.columns(2)

    col_a.metric(
        label=f"Distance montée >= {threshold_up} %",
        value=f"{d_up_km:.2f} km",
        delta=f"{d_up_km / total_dist_km * 100:.1f} % du parcours"
    )
    col_a.metric(
        label=f"D+ cumulé >= {threshold_up} %",
        value=f"{dplus_filtered:.0f} m",
        delta=f"{dplus_filtered / total_pos * 100:.1f} % du D+ total" if total_pos > 0 else "-"
    )

    col_b.metric(
        label=f"Distance descente >= {threshold_down} %",
        value=f"{d_down_km:.2f} km",
        delta=f"{d_down_km / total_dist_km * 100:.1f} % du parcours"
    )
    col_b.metric(
        label=f"D- cumulé >= {threshold_down} %",
        value=f"{dminus_filtered:.0f} m",
        delta=f"{dminus_filtered / total_neg * 100:.1f} % du D- total" if total_neg > 0 else "-"
    )


def render_tab3(distances, elevations, seg_gradient, seg_dist, seg_mid, total_dist_km):
    """
    Affiche le contenu de l'onglet 3 : analyse d'un tronçon sélectionnable.

    Paramètres d'entrée :
        distances     (np.ndarray) : Distances cumulées GPS en mètres.
        elevations    (np.ndarray) : Altitudes GPS en mètres.
        seg_gradient  (np.ndarray) : Gradient de chaque segment en %.
        seg_dist      (np.ndarray) : Longueur de chaque segment en mètres.
        seg_mid       (np.ndarray) : Distance du milieu de chaque segment en mètres.
        total_dist_km (float)      : Distance totale en km.
    """
    # Sliders de début et de fin de tronçon.
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        troncon_start = st.slider(
            "Début du tronçon (km)",
            min_value=0.0,
            max_value=float(round(total_dist_km, 2)),
            value=0.0,
            step=0.1
        )
    with col_t2:
        troncon_end = st.slider(
            "Fin du tronçon (km)",
            min_value=0.0,
            max_value=float(round(total_dist_km, 2)),
            value=float(round(total_dist_km / 2, 1)),
            step=0.1
        )

    if troncon_end <= troncon_start:
        st.warning("La fin du tronçon doit être supérieure au début.")
        st.stop()

    start_m = troncon_start * 1000
    end_m   = troncon_end   * 1000

    # Distance du tronçon (somme des segments dont le milieu est dans le tronçon).
    mask_troncon    = (seg_mid >= start_m) & (seg_mid <= end_m)
    troncon_dist_km = seg_dist[mask_troncon].sum() / 1000

    # Statistiques du tronçon.
    troncon_dplus, troncon_dminus, troncon_gradient_moyen = compute_troncon_stats(
        seg_gradient, seg_dist, seg_mid,
        distances, elevations,
        start_m, end_m, troncon_dist_km
    )

    # Graphique du tronçon.
    fig_troncon = build_troncon_figure(distances, elevations, troncon_start, troncon_end)
    st.plotly_chart(fig_troncon, use_container_width=True)

    # Métriques du tronçon.
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Distance",        f"{troncon_dist_km:.2f} km")
    col_m2.metric("D+",              f"{troncon_dplus:.0f} m")
    col_m3.metric("D−",              f"{troncon_dminus:.0f} m")
    col_m4.metric("Gradient moyen",  f"{troncon_gradient_moyen:+.1f} %")


def render_tab4(distances, elevations, seg_start, seg_end, seg_dist,
                seg_gradient, seg_surface_key, points, total_dist_km):
    """
    Affiche le contenu de l'onglet 4 : type de surface (OSM + correction manuelle).

    Paramètres d'entrée :
        distances       (np.ndarray) : Distances cumulées GPS en mètres.
        elevations      (np.ndarray) : Altitudes GPS en mètres.
        seg_start       (np.ndarray) : Distances de début de chaque segment en mètres.
        seg_end         (np.ndarray) : Distances de fin de chaque segment en mètres.
        seg_dist        (np.ndarray) : Longueur de chaque segment en mètres.
        seg_gradient    (np.ndarray) : Gradient de chaque segment en %.
        seg_surface_key (str)        : Clé du session_state pour la liste de surfaces.
        points          (list)       : Points GPS bruts (lat, lon, elev).
        total_dist_km   (float)      : Distance totale en km.
    """
    st.markdown("### Type de surface")

    # Initialisation du session_state si absent.
    if seg_surface_key not in st.session_state:
        st.session_state[seg_surface_key] = ["Inconnu"] * len(seg_start)

    # Bouton de détection automatique via OSM.
    col_osm1, col_osm2 = st.columns([3, 1])
    with col_osm1:
        st.caption("Détection automatique via OpenStreetMap (Overpass API)")
    with col_osm2:
        if st.button("Détecter automatiquement", key="btn_osm"):
            with st.spinner("Interrogation OpenStreetMap..."):
                st.session_state[seg_surface_key] = fetch_surface_from_osm(
                    points, seg_start, seg_dist
                )
            st.success("Détection terminée.")

    # Correction manuelle d'un tronçon.
    with st.expander("Corriger manuellement un tronçon"):
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            m_start = st.number_input("Début (km)", min_value=0.0,
                                       max_value=float(total_dist_km), value=0.0, step=0.1)
        with col_m2:
            m_end   = st.number_input("Fin (km)", min_value=0.0,
                                       max_value=float(total_dist_km),
                                       value=float(total_dist_km), step=0.1)
        with col_m3:
            m_type  = st.selectbox("Type de surface", SURFACE_TYPES)
        with col_m4:
            st.write("")
            st.write("")
            if st.button("Appliquer", key="btn_manual_surface"):
                surf = st.session_state[seg_surface_key].copy()
                for i in range(len(seg_start)):
                    if m_start <= seg_start[i] / 1000 < m_end:
                        surf[i] = m_type
                st.session_state[seg_surface_key] = surf
                st.rerun()

    # Graphique de profil coloré par surface.
    fig_surface = build_surface_figure(
        distances, elevations,
        seg_start, seg_end,
        st.session_state[seg_surface_key]
    )
    st.plotly_chart(fig_surface, use_container_width=True)

    # Tableau récapitulatif des tronçons.
    df_surface = build_surface_table(seg_start, seg_dist, st.session_state[seg_surface_key])
    st.dataframe(df_surface, use_container_width=True, hide_index=True)

    # Métriques résumées par type de surface.
    st.markdown("**Résumé :**")
    total_dist_m = seg_dist.sum()
    cols_surf    = st.columns(len(SURFACE_TYPES))

    for j, surf_type in enumerate(SURFACE_TYPES):
        mask      = [s == surf_type for s in st.session_state[seg_surface_key]]
        dist_surf = seg_dist[mask].sum() / 1000
        pct_surf  = dist_surf / (total_dist_m / 1000) * 100
        cols_surf[j].metric(surf_type, f"{dist_surf:.1f} km", f"{pct_surf:.1f} %")


def render_map_section(points):
    """
    Affiche la carte Folium interactive du tracé GPX.

    Le paramètre returned_objects=[] est essentiel pour la fluidité : il empêche
    Streamlit de déclencher un rechargement côté serveur à chaque interaction
    (zoom, déplacement) de la carte. Sans ce paramètre, chaque interaction envoie
    un message au serveur Python et provoque un re-run partiel de l'application,
    d'où le délai visible. Avec returned_objects=[], la carte devient purement
    client-side après son premier rendu.

    Paramètres d'entrée :
        points (list of tuples) : Points GPS bruts (lat, lon, elev).
    """
    m = build_route_map(points)

    # returned_objects=[] : aucun callback serveur lors des interactions carte.
    st_folium(m, width="100%", height=600, returned_objects=[])



