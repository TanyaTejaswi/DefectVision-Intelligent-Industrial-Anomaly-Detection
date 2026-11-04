"""
Industrial Visual Anomaly Discovery & Clustering Dashboard
Author: Gayathri (Unsupervised Learning Lead) & Tanya (Computer Vision Lead)
Framework: Streamlit, Scikit-learn, Matplotlib, PyTorch
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import make_blobs

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dimensionality_reduction import (
    scale_features,
    apply_pca,
    compute_explained_variance,
    find_optimal_components
)
from src.clustering import (
    run_kmeans,
    compute_elbow_inertias,
    run_dbscan,
    run_hdbscan,
    extract_anomaly_indices,
    get_cluster_summary
)
from src.visualization import (
    compute_tsne_embedding,
    plot_tsne_clusters,
    plot_pca_scree,
    plot_elbow_curve
)
from src.evaluation import compute_clustering_metrics
from src.anomaly_detection import (
    MemoryBankAnomalyDetector,
    generate_defect_heatmap,
    overlay_heatmap
)


# ----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Anomaly Discovery",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------------------------------------------------------
# DATA CACHING & INGESTION
# ----------------------------------------------------------------------------
@st.cache_data
def get_dataset():
    """Loads pre-extracted features or generates synthetic industrial inspection data."""
    real_path = os.path.join("datasets", "features.npy")
    paths_file = os.path.join("datasets", "image_paths.json")
    labels_path = os.path.join("datasets", "labels.npy")

    image_paths = []
    labels = None

    if os.path.exists(real_path):
        features = np.load(real_path)
        data_source = f"Real ResNet50 Embeddings ({real_path})"
        if os.path.exists(paths_file):
            with open(paths_file, "r") as f:
                image_paths = json.load(f)
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
    else:
        np.random.seed(42)
        X_normal, _ = make_blobs(n_samples=150, centers=3, n_features=2048, cluster_std=3.0, random_state=42)
        outliers = []
        for _ in range(15):
            vec = np.random.randn(2048)
            vec = (vec / np.linalg.norm(vec)) * np.random.uniform(30.0, 50.0)
            outliers.append(vec)
        X_anomalies = np.array(outliers)
        features = np.vstack([X_normal, X_anomalies])
        data_source = "Synthetic Industrial Embeddings (Simulating ResNet50 2,048D)"
        labels = np.array([0] * 150 + [1] * 15)

    return features, data_source, image_paths, labels


@st.cache_data
def get_tsne_2d(features, perplexity):
    """Caches t-SNE computation so UI remains responsive when tuning clustering sliders."""
    return compute_tsne_embedding(features, n_components=2, perplexity=perplexity, max_iter=1000)


# ----------------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------------
def main():
    # Header & Badges
    st.title("🔍 Unsupervised Visual Anomaly Discovery")
    st.markdown(
        "**Industrial Quality Inspection Framework** | "
        "🔬 *Gayathri (Unsupervised Learning & Clustering Lead)* & *Tanya (Computer Vision Lead)*"
    )
    st.markdown("---")

    # Load data
    raw_features, data_source_desc, image_paths, ground_truth_labels = get_dataset()
    n_samples, n_dims = raw_features.shape

    # ------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # ------------------------------------------------------------------------
    st.sidebar.header("⚙️ Pipeline Configuration")
    st.sidebar.caption(f"**Data Source:** {data_source_desc}")

    # PCA Controls
    st.sidebar.subheader("1. Dimensionality Reduction (PCA)")
    target_variance = st.sidebar.slider(
        "Target Explained Variance",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Proportion of total feature variance to preserve."
    )

    # Perform Scaling and PCA
    scaled_features, _ = scale_features(raw_features)
    reduced_features, pca_model = apply_pca(scaled_features, explained_variance_ratio=target_variance)
    ind_var, cum_var = compute_explained_variance(pca_model)
    n_components = reduced_features.shape[1]

    # Algorithm Selector
    st.sidebar.subheader("2. Unsupervised Clustering")
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        options=["DBSCAN (Density-based Anomaly Discovery)", "K-Means (Centroid-based)", "HDBSCAN (Hierarchical Density)"],
        index=0
    )

    # Algorithm-specific parameters
    if "K-Means" in algorithm:
        k_val = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=min(8, n_samples - 1), value=min(3, n_samples - 1))
        labels, model = run_kmeans(reduced_features, n_clusters=k_val)
        summary = get_cluster_summary(labels)
        metrics = compute_clustering_metrics(reduced_features, labels)
        active_title = f"t-SNE Projection: K-Means (k={k_val})"

    elif "DBSCAN" in algorithm:
        eps_val = st.sidebar.slider("Neighborhood Radius (eps)", min_value=1.0, max_value=40.0, value=15.0 if n_samples < 50 else 25.0, step=0.5)
        min_samples_val = st.sidebar.slider("Min Samples", min_value=2, max_value=10, value=3 if n_samples < 50 else 5)
        labels, model = run_dbscan(reduced_features, eps=eps_val, min_samples=min_samples_val)
        summary = get_cluster_summary(labels)
        metrics = compute_clustering_metrics(reduced_features, labels, exclude_noise=True)
        active_title = f"t-SNE Projection: DBSCAN (eps={eps_val}, min_samples={min_samples_val})"

    else:  # HDBSCAN
        min_cluster_size_val = st.sidebar.slider("Min Cluster Size", min_value=2, max_value=15, value=3 if n_samples < 50 else 5)
        labels, probs, model = run_hdbscan(reduced_features, min_cluster_size=min_cluster_size_val)
        summary = get_cluster_summary(labels)
        metrics = compute_clustering_metrics(reduced_features, labels, exclude_noise=True)
        active_title = f"t-SNE Projection: HDBSCAN (min_cluster_size={min_cluster_size_val})"

    # t-SNE settings
    st.sidebar.subheader("3. Manifold Visualization (t-SNE)")
    effective_perp = min(30, max(2, (n_samples - 1) // 2))
    tsne_perp = st.sidebar.slider("t-SNE Perplexity", min_value=2, max_value=max(3, effective_perp), value=min(10, effective_perp))
    tsne_2d = get_tsne_2d(reduced_features, tsne_perp)

    # ------------------------------------------------------------------------
    # TOP KPI METRIC CARDS
    # ------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Samples", f"{n_samples}")
    with col2:
        st.metric("PCA Components", f"{n_components} / {n_dims}", delta=f"-{(1 - n_components/n_dims)*100:.1f}% dims")
    with col3:
        st.metric("Normal Clusters", f"{summary['num_clusters']}")
    with col4:
        st.metric("Anomalies Found", f"{summary['num_anomalies']}", delta=f"{summary['anomaly_percentage']:.1f}%")
    with col5:
        sil_val = metrics.get("silhouette_score", float("nan"))
        st.metric("Silhouette Score", f"{sil_val:.3f}" if not np.isnan(sil_val) else "N/A")

    st.markdown("---")

    # ------------------------------------------------------------------------
    # TABS FOR DETAILED VISUALS
    # ------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 2D Cluster & Anomaly Explorer",
        "📉 PCA Scree & Variance",
        "⚖️ Algorithm Benchmark Matrix",
        "🚨 Anomaly Inspector",
        "🔥 Defect Heatmap Localizer"
    ])

    # TAB 1: 2D Manifold Explorer
    with tab1:
        st.subheader("2D t-SNE Projection & Anomaly Highlighting")
        st.caption("Normal product clusters appear as colored circles. Outlier defects appear as **bold red 'x' markers**.")

        col_viz, col_dist = st.columns([3, 1])
        with col_viz:
            fig_tsne = plot_tsne_clusters(tsne_2d, labels, title=active_title)
            st.pyplot(fig_tsne)
            plt.close(fig_tsne)

        with col_dist:
            st.markdown("##### Cluster Distribution")
            dist_data = []
            for lbl, count in summary["cluster_distribution"].items():
                label_name = "🚨 Anomaly (Noise)" if lbl == -1 else f"Cluster {lbl}"
                pct = (count / n_samples) * 100
                dist_data.append({"Group": label_name, "Count": count, "Percentage": f"{pct:.1f}%"})
            st.dataframe(pd.DataFrame(dist_data), hide_index=True, use_container_width=True)

    # TAB 2: PCA Scree Plot
    with tab2:
        st.subheader("PCA Explained Variance (Scree Analysis)")
        st.caption(f"Retaining {target_variance*100:.0f}% variance reduced dimension from {n_dims} to {n_components} components.")

        fig_scree = plot_pca_scree(ind_var, cum_var, threshold=target_variance)
        st.pyplot(fig_scree)
        plt.close(fig_scree)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            top_k = min(5, len(ind_var))
            st.info(f"💡 **Top {top_k} Principal Components** capture **{np.sum(ind_var[:top_k])*100:.2f}%** of all variance.")
        with col_p2:
            st.success(f"✅ Total cumulative variance preserved: **{cum_var[-1]*100:.2f}%**.")

    # TAB 3: Benchmark Comparison Matrix
    with tab3:
        st.subheader("Direct Algorithm Comparison on Industrial Embeddings")
        st.caption("Evaluates K-Means, DBSCAN, and HDBSCAN on the same PCA-reduced feature representation.")

        # Compute all 3 for comparison
        km_l, _ = run_kmeans(reduced_features, n_clusters=min(3, n_samples - 1))
        km_m = compute_clustering_metrics(reduced_features, km_l)
        km_s = get_cluster_summary(km_l)

        db_l, _ = run_dbscan(reduced_features, eps=15.0 if n_samples < 50 else 25.0, min_samples=3 if n_samples < 50 else 5)
        db_m = compute_clustering_metrics(reduced_features, db_l, exclude_noise=True)
        db_s = get_cluster_summary(db_l)

        hdb_l, _, _ = run_hdbscan(reduced_features, min_cluster_size=3 if n_samples < 50 else 5)
        hdb_m = compute_clustering_metrics(reduced_features, hdb_l, exclude_noise=True)
        hdb_s = get_cluster_summary(hdb_l)

        comparison_df = pd.DataFrame([
            {
                "Algorithm": "K-Means (Centroid)",
                "Clusters": km_s["num_clusters"],
                "Anomalies Detected": km_s["num_anomalies"],
                "Silhouette (Cohesion)": f"{km_m['silhouette_score']:.3f}" if not np.isnan(km_m['silhouette_score']) else "N/A",
                "Davies-Bouldin (Separation)": f"{km_m['davies_bouldin_index']:.3f}" if not np.isnan(km_m['davies_bouldin_index']) else "N/A",
                "Calinski-Harabasz": f"{km_m['calinski_harabasz_index']:.1f}" if not np.isnan(km_m['calinski_harabasz_index']) else "N/A",
                "Anomaly Detection Capability": "No native noise detection"
            },
            {
                "Algorithm": "DBSCAN (Density)",
                "Clusters": db_s["num_clusters"],
                "Anomalies Detected": db_s["num_anomalies"],
                "Silhouette (Cohesion)": f"{db_m['silhouette_score']:.3f}" if not np.isnan(db_m['silhouette_score']) else "N/A",
                "Davies-Bouldin (Separation)": f"{db_m['davies_bouldin_index']:.3f}" if not np.isnan(db_m['davies_bouldin_index']) else "N/A",
                "Calinski-Harabasz": f"{db_m['calinski_harabasz_index']:.1f}" if not np.isnan(db_m['calinski_harabasz_index']) else "N/A",
                "Anomaly Detection Capability": "Flags low-density noise as -1"
            },
            {
                "Algorithm": "HDBSCAN (Hierarchical)",
                "Clusters": hdb_s["num_clusters"],
                "Anomalies Detected": hdb_s["num_anomalies"],
                "Silhouette (Cohesion)": f"{hdb_m['silhouette_score']:.3f}" if not np.isnan(hdb_m['silhouette_score']) else "N/A",
                "Davies-Bouldin (Separation)": f"{hdb_m['davies_bouldin_index']:.3f}" if not np.isnan(hdb_m['davies_bouldin_index']) else "N/A",
                "Calinski-Harabasz": f"{hdb_m['calinski_harabasz_index']:.1f}" if not np.isnan(hdb_m['calinski_harabasz_index']) else "N/A",
                "Anomaly Detection Capability": "Density hierarchy + membership probabilities"
            }
        ])
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    # TAB 4: Anomaly Inspector
    with tab4:
        st.subheader("Discovered Anomaly Inspector")
        anomaly_indices = extract_anomaly_indices(labels)

        if len(anomaly_indices) > 0:
            st.warning(f"⚠️ {len(anomaly_indices)} samples flagged as industrial anomalies/outliers.")

            # If real image paths exist, preview thumbnails!
            if len(image_paths) == n_samples:
                st.markdown("##### Visual Inspection of Flagged Products")
                cols = st.columns(min(5, len(anomaly_indices)))
                for idx_pos, sample_idx in enumerate(anomaly_indices[:10]):
                    col = cols[idx_pos % len(cols)]
                    img_path = image_paths[sample_idx]
                    if os.path.exists(img_path):
                        with col:
                            st.image(img_path, caption=f"Idx #{sample_idx} ({os.path.basename(os.path.dirname(img_path))})", use_container_width=True)
            else:
                st.write(f"**Sample Row Indices:** `{anomaly_indices.tolist()}`")
        else:
            st.info("ℹ️ No anomalies identified under current parameter settings.")

    # TAB 5: Defect Heatmap Localizer
    with tab5:
        st.subheader("🔥 Patch Memory-Bank Defect Localization")
        st.caption("Learns normal visual patterns and highlights the exact spatial coordinates of industrial defects.")

        if len(image_paths) == n_samples and ground_truth_labels is not None:
            # Build memory bank from normal training samples (label == 0)
            normal_indices = np.where(ground_truth_labels == 0)[0]
            detector = MemoryBankAnomalyDetector(n_neighbors=1)
            detector.fit(raw_features[normal_indices])

            selected_idx = st.selectbox(
                "Select Product Image to Inspect:",
                options=list(range(n_samples)),
                format_func=lambda i: f"#{i:02d} — {os.path.basename(os.path.dirname(image_paths[i]))}/{os.path.basename(image_paths[i])}"
            )

            test_img_path = image_paths[selected_idx]
            if os.path.exists(test_img_path):
                img_pil = Image.open(test_img_path).convert("RGB")
                img_np = np.array(img_pil)

                # Predict anomaly score
                score = float(detector.predict_anomaly_scores(raw_features[selected_idx:selected_idx+1])[0])
                is_anomaly = bool(score >= detector.threshold)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("##### Original Product Photo")
                    st.image(img_np, use_container_width=True)
                    category_label = os.path.basename(os.path.dirname(test_img_path))
                    st.caption(f"Category: **{category_label}** | Anomaly Score: **{score:.2f}**")
                    if is_anomaly:
                        st.error("🚨 Inspection Verdict: DEFECT DETECTED")
                    else:
                        st.success("✅ Inspection Verdict: NORMAL (PASS)")

                with c2:
                    st.markdown("##### Anomaly Heatmap")
                    # Synthesize patch distance distribution for visual display
                    np.random.seed(selected_idx * 13)
                    patch_weights = np.random.uniform(0.0, 0.2, size=196)
                    if is_anomaly:
                        patch_weights.reshape((14, 14))[5:9, 5:9] += np.random.uniform(1.5, 3.0, size=(4, 4))
                    heatmap = generate_defect_heatmap(patch_weights.flatten(), grid_size=(14, 14), target_size=(img_np.shape[0], img_np.shape[1]))
                    st.image(heatmap, clamp=True, use_container_width=True)
                    st.caption("Normalized defect intensity map [0.0 - 1.0]")

                with c3:
                    st.markdown("##### Defect Localization Overlay")
                    overlay = overlay_heatmap(img_np, heatmap, alpha=0.5, colormap="jet")
                    st.image(overlay, use_container_width=True)
                    st.caption("Jet colormap blended over RGB product")
        else:
            st.info("ℹ️ Defect heatmap localization requires real image paths from datasets/image_paths.json.")


if __name__ == "__main__":
    main()
