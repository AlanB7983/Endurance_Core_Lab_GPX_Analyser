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

import streamlit as st
import subprocess
import sys

# --- Installation dynamique du package privé (fonctions propriétaires) ---
# Nécessaire car Streamlit Cloud n'expanse pas les variables d'environnement
# dans requirements.txt au moment du build. On installe donc le package
# manuellement ici, au runtime, où st.secrets est disponible.
def install_private_package():
    try:
        import gpx_functions  # noqa: F401
    except ImportError:
        token = st.secrets["GITHUB_TOKEN"]
        repo_url = (
            f"git+https://{token}@github.com/"
            "AlanB7983/Endurance_Core_Lab_GPX_Analyser_Private.git"
        )
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", repo_url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            st.error("Échec de l'installation du package privé :")
            st.code(result.stdout + "\n" + result.stderr)
            st.stop()

install_private_package()




from gpw_functions.package import parse_gpx_file, compute_cumulative_distances_and_elevations, discretize_into_100m_segments, fetch_surface_from_osm, compute_global_stats, render_tab1, render_tab2, render_tab3, render_tab4, render_synced_map_section, render_map_section
#from gpx_package import parse_gpx_file, compute_cumulative_distances_and_elevations, discretize_into_100m_segments, fetch_surface_from_osm, compute_global_stats, render_tab1, render_tab2, render_tab3, render_tab4, render_synced_map_section, render_map_section




# =============================================================================
# SECTION 11 — POINT D'ENTRÉE : FONCTION MAIN
# =============================================================================

def main():
    """
    Point d'entrée de l'application Streamlit.

    Orchestre dans l'ordre :
        1. Configuration de la page.
        2. Chargement du fichier GPX.
        3. Parsing et calculs préliminaires (distances, segments, stats globales).
        4. Affichage des onglets d'analyse.
        5. Affichage de la carte interactive.
    """
    # ── Configuration de la page ─────────────────────────────────────────────
    st.set_page_config(
        page_title="Trail GPX Analyzer",
        layout="wide",
        page_icon="Pictures/__LOGO_ECICONE_NOIR.png"
    )
    
    col1_logo, col2_logo, col3_logo = st.columns([1, 2, 1])  # Ajuste les proportions

    with col1_logo:
        st.image("Pictures/__LOGO_ECLAB_ORANGE.png", use_container_width=True)

    st.header("ANALYSE GPX")
    st.markdown("---")
    
    with st.expander("Comment fonctionne cet outil ?", expanded=False):
        st.markdown("""
        Cet outil analyse le tracé GPX de votre course ou sortie trail et en tire un profil détaillé, exploitable pour la préparation comme pour le débriefing.
    
        **1. Profil altimétrique et gradients :**
        Le parcours est découpé en tronçons de 100 m. Chaque tronçon est coloré selon sa pente (du jaune clair pour le plat au rouge brun pour les fortes montées, du vert clair au vert foncé pour les descentes). Un tableau récapitule la distance et le dénivelé parcourus dans chaque tranche de pente.
    
        **2. Visualisation filtrée :**
        Deux curseurs permettent d'isoler uniquement les montées et descentes les plus raides, pour repérer rapidement les passages clés du parcours et savoir par exemple si les bâtons sont nécessaires et à quels endroits.
    
        **3. Analyse d'un tronçon :**
        Sélectionnez une portion du parcours (par exemple une montée spécifique) pour en obtenir la distance, le D+, le D- et le gradient moyen.
    
        **4. Type de surface :**
        Le parcours est découpé en tronçons de 100 m. L'outil détecte automatiquement la nature du terrain (route, gravel, sentier technique) en croisant le tracé avec les données OpenStreetMap pour chaque tronçon.
    
        **5. Carte interactive :**
        Le tracé est affiché sur une carte que vous pouvez zoomer et déplacer librement. En survolant le profil altimétrique, un repère se déplace sur la carte à la position correspondante, pour visualiser en un coup d'œil où vous en êtes sur le parcours.
    
        *Pour commencer, chargez simplement votre fichier GPX ci-dessous.*
        """)

    # ── Étape 1 : chargement du fichier GPX ─────────────────────────────────
    uploaded_file = st.file_uploader("Charger un fichier GPX", type=["gpx"])

    if not uploaded_file:
        st.info("Chargez un fichier GPX pour commencer l'analyse.")
        st.markdown("""
        ### Ce que cet outil analyse :
        - **Profil altimétrique** coloré par tranche de gradient (résolution 100 m)
        - **Tableau statistique** : distance et dénivelé par tranche de gradient (montée & descente)
        - **Visualisation filtrée** : mettez en évidence les portions les plus raides
        - **Détection du type de surface** : détectez le type de surface pour évaluer la technicité du parcours
        """)
        return

    # ── Étape 2 : parsing et calculs préliminaires ───────────────────────────
    try:
        points = parse_gpx_file(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier GPX : {e}")
        return

    if len(points) < 2:
        st.error("Le fichier GPX ne contient pas assez de points (minimum 2 requis).")
        return

    # Distances cumulées et altitudes brutes.
    distances, elevations = compute_cumulative_distances_and_elevations(points)

    # Discrétisation en segments de 100 m avec gradients.
    bp, elev_bp, seg_start, seg_end, seg_mid, seg_dist, seg_gradient = (
        discretize_into_100m_segments(distances, elevations)
    )

    # Détection automatique de surface au premier chargement du fichier.
    # Le nom du fichier sert d'identifiant : si un nouveau fichier est chargé,
    # la détection est relancée ; sinon on conserve le résultat en session.
    gpx_file_id      = uploaded_file.name
    SURFACE_STATE_KEY = "seg_surface"

    if (
        SURFACE_STATE_KEY not in st.session_state
        or st.session_state.get("seg_surface_file_id") != gpx_file_id
    ):
        with st.spinner("Détection automatique du type de surface (OSM)..."):
            st.session_state[SURFACE_STATE_KEY]     = fetch_surface_from_osm(
                points, seg_start, seg_dist
            )
            st.session_state["seg_surface_file_id"] = gpx_file_id

    # Statistiques globales du parcours.
    total_dist_km, total_pos, total_neg = compute_global_stats(distances, elev_bp)

    # Bandeau de résumé en haut de page.
    st.success(
        f"Fichier chargé — {len(points)} points GPS | "
        f"Distance : **{total_dist_km:.2f} km** | "
        f"D+ : **{total_pos:.0f} m** | "
        f"D- : **{total_neg:.0f} m**"
    )
    
    st.markdown("---")

    # ── Étape 3 : onglets d'analyse ──────────────────────────────────────────
    st.markdown("### Analyse du profil")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Profil altimétrique avec gradients et statistiques associées",
        "Visualisation filtrée par gradient",
        "Analyse d'un tronçon",
        "Type de surface"
    ])

    with tab1:
        render_tab1(
            distances, elevations,
            seg_start, seg_end, seg_dist, seg_gradient,
            elev_bp, total_pos, total_neg, total_dist_km
        )

    with tab2:
        render_tab2(
            distances, elevations,
            seg_start, seg_end, seg_dist, seg_gradient,
            total_dist_km, total_pos, total_neg
        )

    with tab3:
        render_tab3(
            distances, elevations,
            seg_gradient, seg_dist, seg_mid, total_dist_km
        )

    with tab4:
        render_tab4(
            distances, elevations,
            seg_start, seg_end, seg_dist, seg_gradient,
            SURFACE_STATE_KEY, points, total_dist_km
        )

    # ── Étape 4 : carte interactive ──────────────────────────────────────────
    st.markdown("---")
    
    st.markdown("### Carte intéractive")

    # Choix entre la carte synchronisée (curseur lié au profil) et la carte
    # Folium classique (plus légère, sans synchronisation).
    map_mode = st.radio(
        "Mode d'affichage de la carte",
        options=["Carte synchronisée avec le profil", "Carte simple (Folium)"],
        horizontal=True
    )

    if map_mode == "Carte synchronisée avec le profil":
        render_synced_map_section(points, distances, elevations)
    else:
        render_map_section(points)


# =============================================================================
# SECTION 12 — LANCEMENT
# =============================================================================

if __name__ == "__main__":
    main()
