import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# Import các module trong project
from d23_lop4_nhom16.config import get_db_url
from d23_lop4_nhom16.database_loader import load_database_features
from d23_lop4_nhom16.extract_query_features import extract_query_features
from d23_lop4_nhom16.feature_scores import feature_scores
from d23_lop4_nhom16.fusions import normalize_and_ranking
from d23_lop4_nhom16.similarity.glcm_similarity import glcm_scaler

# ==============================================================================
# CẤU HÌNH TRANG
# ==============================================================================
st.set_page_config(
    page_title="CBVR – Truy vấn Video Chó",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = {
    "hsv":  {"label": "HSV",  "color": "#e9a23b"},
    "glcm": {"label": "GLCM", "color": "#8a7fd1"},
    "hog":  {"label": "HOG",  "color": "#3a9ad9"},
    "flow": {"label": "Flow", "color": "#2bb39a"},
    "mhi":  {"label": "MHI",  "color": "#e0607e"},
}

PRESETS = {
    "Cân bằng (mặc định)": (0.15, 0.10, 0.20, 0.25, 0.30),
    "Ưu tiên hành động":   (0.05, 0.05, 0.10, 0.35, 0.45),
    "Ưu tiên ngoại hình":  (0.35, 0.15, 0.35, 0.10, 0.05),
}

CARD_MIN_WIDTH = 270  # px – bề rộng tối thiểu của mỗi thẻ kết quả

# ==============================================================================
# CSS
# - Khung video: ép wrapper về 16:9 (nền đen, video co vừa khung) -> mọi video
#   đầu vào và đầu ra có cùng kích thước, kể cả video quay dọc.
# - Hàng kết quả: tất cả thẻ nằm trên MỘT HÀNG NGANG; nếu nhiều hơn chiều rộng
#   màn hình thì cuộn ngang trong hàng đó.
# ==============================================================================
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1800px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    .app-title {{ font-size: 1.5rem; font-weight: 700; margin: 0; line-height: 1.2; }}
    .app-sub   {{ font-size: .85rem; opacity: .65; margin: .15rem 0 .6rem 0; }}

    /* ---- Khung video đồng nhất 16:9 ---- */
    div[data-testid="stVideo"], .stVideo {{
        width: 100% !important;
        aspect-ratio: 16 / 9 !important;
        background: #000;
        border-radius: 8px;
        overflow: hidden;
    }}
    div[data-testid="stVideo"] video, .stVideo video {{
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
        display: block;
    }}
    .video-empty {{
        width: 100%; aspect-ratio: 16 / 9; border-radius: 8px;
        border: 2px dashed rgba(128,128,128,.4);
        display: flex; align-items: center; justify-content: center;
        font-size: .9rem; opacity: .6; text-align: center;
    }}

    /* ---- Hàng kết quả: 1 hàng ngang, cuộn ngang khi tràn ---- */
    div[data-testid="stHorizontalBlock"]:has(.results-row) {{
        flex-wrap: nowrap !important;
        overflow-x: auto;
        gap: .75rem !important;
        padding-bottom: .6rem;
        align-items: stretch;
    }}
    div[data-testid="stHorizontalBlock"]:has(.results-row) > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"]:has(.results-row) > div[data-testid="column"] {{
        flex: 1 0 {CARD_MIN_WIDTH}px !important;
        min-width: {CARD_MIN_WIDTH}px !important;
        width: auto !important;
    }}
    /* marker không chiếm chỗ */
    div[data-testid="stElementContainer"]:has(.results-row),
    .element-container:has(.results-row) {{ display: none; }}

    /* ---- Thẻ kết quả ---- */
    .card-head {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: .35rem; gap: .5rem;
    }}
    .card-head .rank {{ font-weight: 700; font-size: 1rem; }}
    .card-head .cat  {{ font-size: .85rem; opacity: .75; flex: 1; }}
    .card-head .score {{
        font-weight: 700; font-size: .9rem; padding: 1px 9px; border-radius: 999px;
        background: rgba(43,179,154,.18); color: #2bb39a;
    }}
    .card-file {{ font-size: .75rem; opacity: .6; margin: .3rem 0 .45rem 0;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

    .mini-row {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; font-size: .72rem; }}
    .mini-row .name  {{ width: 38px; opacity: .75; }}
    .mini-row .track {{ flex: 1; height: 6px; border-radius: 3px; background: rgba(128,128,128,.22); overflow: hidden; }}
    .mini-row .fill  {{ height: 100%; border-radius: 3px; }}
    .mini-row .val   {{ width: 34px; text-align: right; font-variant-numeric: tabular-nums; }}

    /* ---- Sidebar: thanh phân bổ trọng số ---- */
    .wbar {{ display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin: .3rem 0 .5rem 0; }}
    .wlegend {{ display: flex; flex-wrap: wrap; gap: 4px 10px; font-size: .75rem; }}
    .wlegend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }}

    /* ---- Bảng markdown ---- */
    div[data-testid="stMarkdownContainer"] table {{ width: 100%; font-size: .85rem; }}
    div[data-testid="stMarkdownContainer"] th {{ text-align: center !important; }}
    div[data-testid="stMarkdownContainer"] td {{ text-align: center !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# CACHE DỮ LIỆU (chỉ tải 1 lần)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_cached_database():
    engine = create_engine(get_db_url())
    df_videos, db_store = load_database_features(engine)
    scaler = glcm_scaler(engine)
    return engine, df_videos, db_store, scaler


# ==============================================================================
# HÀM TIỆN ÍCH
# ==============================================================================
def mini_bars_html(item) -> str:
    rows = []
    for key, meta in FEATURES.items():
        v = float(item.get(f"norm_{key}", 0.0) or 0.0)
        v = max(0.0, min(1.0, v))
        rows.append(
            f"<div class='mini-row'><span class='name'>{meta['label']}</span>"
            f"<div class='track'><div class='fill' style='width:{v*100:.0f}%;background:{meta['color']}'></div></div>"
            f"<span class='val'>{v:.2f}</span></div>"
        )
    return "".join(rows)


def weight_bar_html(weights: dict, total: float) -> str:
    segs, legend = [], []
    for key, meta in FEATURES.items():
        pct = weights[key] / total * 100 if total > 0 else 0
        segs.append(f"<div style='width:{pct:.1f}%;background:{meta['color']}'></div>")
        legend.append(f"<span><i style='background:{meta['color']}'></i>{meta['label']} {pct:.0f}%</span>")
    return f"<div class='wbar'>{''.join(segs)}</div><div class='wlegend'>{''.join(legend)}</div>"


def medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


# ==============================================================================
# SIDEBAR – CẤU HÌNH
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    top_k = st.slider("Số kết quả trả về (Top-K)", 1, 20, 5, 1)

    preset = st.selectbox("Bộ trọng số", [*PRESETS.keys(), "Tùy chỉnh"])
    if preset in PRESETS:
        w_hsv, w_glcm, w_hog, w_flow, w_mhi = PRESETS[preset]
    else:
        w_hsv  = st.slider("HSV – Màu sắc",         0.0, 1.0, 0.15, 0.05)
        w_glcm = st.slider("GLCM – Kết cấu",         0.0, 1.0, 0.10, 0.05)
        w_hog  = st.slider("HOG – Hình dáng",        0.0, 1.0, 0.20, 0.05)
        w_flow = st.slider("Flow – Chuyển động",      0.0, 1.0, 0.25, 0.05)
        w_mhi  = st.slider("MHI – Nhịp hành động",    0.0, 1.0, 0.30, 0.05)

    weights = {"hsv": w_hsv, "glcm": w_glcm, "hog": w_hog, "flow": w_flow, "mhi": w_mhi}
    total_w = sum(weights.values())

    if total_w > 0:
        st.markdown("**Phân bổ trọng số**")
        st.markdown(weight_bar_html(weights, total_w), unsafe_allow_html=True)
    else:
        st.error("Tổng trọng số phải lớn hơn 0.")

    st.divider()
    st.markdown("**🗄️ Cơ sở dữ liệu**")
    db_status_placeholder = st.empty()

# ==============================================================================
# KẾT NỐI DATABASE
# ==============================================================================
try:
    with st.spinner("Đang kết nối cơ sở dữ liệu..."):
        engine, df_videos, db_store, scaler_glcm = get_cached_database()
    db_status_placeholder.success(f"Sẵn sàng · {len(db_store)} video")
except Exception as e:
    db_status_placeholder.error("Không kết nối được")
    st.error(f"**Lỗi kết nối PostgreSQL:** {e}")
    st.info("Kiểm tra lại dịch vụ PostgreSQL và cấu hình `config.py`.")
    st.stop()

# ==============================================================================
# TIÊU ĐỀ
# ==============================================================================
st.markdown(
    "<p class='app-title'>🐕 Truy vấn video chó theo nội dung</p>"
    "<p class='app-sub'>Content-Based Video Retrieval · HSV · GLCM · HOG · Optical Flow · MHI</p>",
    unsafe_allow_html=True,
)

# ==============================================================================
# HÀNG TRÊN: [Chọn video + nút] | [Xem trước] | [Thông tin kết quả]
# ==============================================================================
query_video_path = None

with st.container(border=True):
    c_ctrl, c_prev, c_info = st.columns([1.2, 1.1, 1], gap="large")

    with c_ctrl:
        st.markdown("##### Video truy vấn")
        input_method = st.radio(
            "Nguồn video", ["Video mẫu", "Tải lên"],
            horizontal=True, label_visibility="collapsed",
        )

        if input_method == "Video mẫu":
            sample_dir = Path(__file__).resolve().parent / "video_test"
            sample_videos = sorted(sample_dir.glob("*.mp4")) if sample_dir.is_dir() else []
            if sample_videos:
                query_video_path = st.selectbox(
                    "Chọn video mẫu", options=sample_videos,
                    format_func=lambda p: p.name, label_visibility="collapsed",
                )
            else:
                st.warning("Chưa có video trong thư mục `video_test/`.")
        else:
            uploaded_file = st.file_uploader(
                "Tải video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"],
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                suffix = Path(uploaded_file.name).suffix or ".mp4"
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tfile.write(uploaded_file.getvalue())
                tfile.flush()
                query_video_path = Path(tfile.name)

        start_search = st.button(
            "🔍 Tìm video tương đồng", type="primary", use_container_width=True,
            disabled=(total_w <= 0 or query_video_path is None),
        )

    with c_prev:
        if query_video_path and Path(query_video_path).is_file():
            st.video(str(query_video_path))
            st.caption(f"📁 {Path(query_video_path).name}")
        else:
            st.markdown("<div class='video-empty'>Chưa chọn video</div>", unsafe_allow_html=True)

    info_ph = c_info.empty()  # điền sau khi có kết quả

# ==============================================================================
# XỬ LÝ TÌM KIẾM (lưu vào session_state để đổi tab không mất kết quả)
# ==============================================================================
if start_search:
    norm_weights = {k: v / total_w for k, v in weights.items()}  # tổng = 1
    t0 = time.time()
    with st.status("Đang xử lý...", expanded=True) as status:
        st.write("🎬 **[1/3]** Trích xuất đặc trưng video truy vấn...")
        q_feat = extract_query_features(query_video_path)

        st.write(f"⚡ **[2/3]** So khớp với {len(db_store)} video trong cơ sở dữ liệu...")
        raw_results = []
        prog = st.progress(0)
        for idx, (vid, c_feat) in enumerate(db_store.items()):
            scores = feature_scores(q_feat, c_feat, scaler_glcm=scaler_glcm)
            scores["video_id"] = vid
            raw_results.append(scores)
            prog.progress((idx + 1) / len(db_store))

        st.write("📊 **[3/3]** Chuẩn hóa Min-Max và xếp hạng theo trọng số...")
        top_results = normalize_and_ranking(
            raw_scores=pd.DataFrame(raw_results),
            df_videos=df_videos,
            weights=norm_weights,
            top_k=top_k,
        )
        status.update(label="Tìm kiếm hoàn tất", state="complete", expanded=False)

    st.session_state["search"] = {
        "top": top_results,
        "elapsed": time.time() - t0,
        "query_name": Path(query_video_path).name,
    }

search = st.session_state.get("search")

# ==============================================================================
# KẾT QUẢ
# ==============================================================================
if search is None:
    info_ph.caption("Kết quả sẽ hiển thị bên dưới sau khi tìm kiếm.")
else:
    top_results = search["top"]

    with info_ph.container():
        st.metric("Số kết quả", f"Top {len(top_results)}")
        st.metric("Thời gian xử lý", f"{search['elapsed']:.2f} s")

    tab_grid, tab_table = st.tabs(["🏆 Kết quả", "📋 Bảng điểm"])

    # ---------------- TAB 1: TẤT CẢ VIDEO TRÊN MỘT HÀNG NGANG ----------------
    with tab_grid:
        cols = st.columns(len(top_results), gap="small")

        for i, (_, item) in enumerate(top_results.iterrows()):
            rank = i + 1
            score_pct = float(item["final_score"]) * 100.0
            category = str(item.get("category", "?")).capitalize()
            cand_path = item.get("filepath", "")

            with cols[i]:
                with st.container(border=True):
                    if i == 0:
                        st.markdown("<span class='results-row'></span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card-head'>"
                        f"<span class='rank'>{medal(rank)}</span>"
                        f"<span class='cat'>{category}</span>"
                        f"<span class='score'>{score_pct:.1f}%</span></div>",
                        unsafe_allow_html=True,
                    )
                    if cand_path and Path(cand_path).is_file():
                        st.video(str(cand_path))
                    else:
                        st.markdown("<div class='video-empty'>Không tìm thấy file</div>",
                                    unsafe_allow_html=True)
                    name = Path(cand_path).name if cand_path else "N/A"
                    st.markdown(f"<div class='card-file'>📁 {name}</div>", unsafe_allow_html=True)
                    st.markdown(mini_bars_html(item), unsafe_allow_html=True)

    # ---------------- TAB 2: BẢNG TỔNG HỢP ----------------
    with tab_table:
        disp_cols = ["video_id", "category", "final_score",
                     "norm_hsv", "norm_glcm", "norm_hog", "norm_flow", "norm_mhi"]
        tbl = top_results[[c for c in disp_cols if c in top_results.columns]].copy()
        for col in ["final_score", "norm_hsv", "norm_glcm", "norm_hog", "norm_flow", "norm_mhi"]:
            if col in tbl.columns:
                tbl[col] = tbl[col].apply(lambda x: f"{float(x):.3f}")

        tbl = tbl.rename(columns={
            "video_id": "ID", "category": "Thể loại", "final_score": "Điểm tổng",
            "norm_hsv": "HSV", "norm_glcm": "GLCM", "norm_hog": "HOG",
            "norm_flow": "Flow", "norm_mhi": "MHI",
        })

        md_cols = list(tbl.columns)
        header = "| " + " | ".join(md_cols) + " |"
        separator = "| " + " | ".join([":---:"] * len(md_cols)) + " |"
        body = "\n".join(
            "| " + " | ".join(str(tbl.iloc[i][c]) for c in md_cols) + " |"
            for i in range(len(tbl))
        )
        st.markdown(f"{header}\n{separator}\n{body}")