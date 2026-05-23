"""Dashboard Streamlit untuk monitoring deteksi merokok.

Membaca event-log JSONL (`runs/events.jsonl`) yang dihasilkan oleh
`alarm.py` saat event ditutup, serta snapshot di `runs/alarm/`.

Cara jalanin:
    pip install streamlit pandas
    streamlit run dashboard.py

Variabel env (opsional):
    EVENT_LOG_PATH      default: runs/events.jsonl
    EVENT_SNAPSHOT_DIR  default: runs/alarm
    LABEL_LOG_PATH      default: runs/labels.jsonl
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


EVENT_LOG = Path(os.getenv("EVENT_LOG_PATH", "runs/events.jsonl"))
SNAPSHOT_DIR = Path(os.getenv("EVENT_SNAPSHOT_DIR", "runs/alarm"))
LABEL_LOG = Path(os.getenv("LABEL_LOG_PATH", "runs/labels.jsonl"))


# ----------------------------------------------------------------- helpers
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return rows


@st.cache_data(ttl=5)
def load_events() -> pd.DataFrame:
    rows = _read_jsonl(EVENT_LOG)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["started_at_dt"] = pd.to_datetime(df["started_at"], unit="s")
    df["ended_at_dt"] = pd.to_datetime(df["ended_at"], unit="s")

    # Gabungkan label false-positive (label terakhir per event_id menang).
    labels = {}
    for r in _read_jsonl(LABEL_LOG):
        eid = r.get("event_id")
        if eid:
            labels[eid] = r.get("is_false_positive")
    df["is_false_positive"] = df["event_id"].map(labels)
    return df


def write_label(event_id: str, is_fp) -> None:
    """Append label baru ke labels.jsonl (label terbaru menang)."""
    LABEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "event_id": event_id,
        "is_false_positive": is_fp,
        "labeled_at": datetime.now().timestamp(),
    }
    with LABEL_LOG.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec) + "\n")
    st.cache_data.clear()


def fmt_ts(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime(fmt)
    except Exception:
        return "-"


# --------------------------------------------------------------------- UI
st.set_page_config(
    page_title="Dashboard Deteksi Merokok",
    page_icon="🚬",
    layout="wide",
)

st.title("🚬 Dashboard Deteksi Merokok")
st.caption(
    f"Sumber data: `{EVENT_LOG}` · Snapshot: `{SNAPSHOT_DIR}` · "
    f"Label: `{LABEL_LOG}`"
)

df = load_events()
if df.empty:
    st.info(
        f"Belum ada event tercatat di `{EVENT_LOG}`. "
        "Jalankan `video.py` untuk mulai merekam event."
    )
    st.stop()

# ---------------- Sidebar: filter & refresh ----------------
with st.sidebar:
    st.header("Filter")
    if st.button("🔄 Muat ulang data"):
        st.cache_data.clear()
        st.rerun()

    min_date = df["started_at_dt"].min().date()
    max_date = df["started_at_dt"].max().date()
    date_range = st.date_input(
        "Rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    available_levels = sorted(df["level"].dropna().unique().tolist())
    levels_sel = st.multiselect(
        "Level", available_levels, default=available_levels
    )

    available_classes = sorted(df["peak_class"].dropna().unique().tolist())
    classes_sel = st.multiselect(
        "Kelas peak", available_classes, default=available_classes
    )

    fp_filter = st.radio(
        "Label",
        ["Semua", "Belum dilabel", "True Positive", "False Positive"],
        index=0,
    )

# ---------------- Apply filter ----------------
mask = df["level"].isin(levels_sel) & df["peak_class"].isin(classes_sel)
if isinstance(date_range, tuple) and len(date_range) == 2:
    mask &= df["started_at_dt"].dt.date.between(date_range[0], date_range[1])
elif hasattr(date_range, "year"):  # single date
    mask &= df["started_at_dt"].dt.date == date_range

if fp_filter == "Belum dilabel":
    mask &= df["is_false_positive"].isna()
elif fp_filter == "True Positive":
    mask &= df["is_false_positive"] == False  # noqa: E712
elif fp_filter == "False Positive":
    mask &= df["is_false_positive"] == True   # noqa: E712

df_f = df[mask].copy()

# ---------------- Metrik ringkas ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Event", len(df_f))
c2.metric("Confirm", int((df_f["level"] == "confirm").sum()))
c3.metric("Suspect", int((df_f["level"] == "suspect").sum()))
c4.metric(
    "Rata-rata Durasi",
    f"{df_f['duration_sec'].mean():.1f}s" if len(df_f) else "-",
)
fp_count = int((df_f["is_false_positive"] == True).sum())  # noqa: E712
c5.metric("False Positive (dilabel)", fp_count)

# ---------------- Metrik reduksi notifikasi (kontribusi sistem) -----------
st.subheader("📉 Reduksi Notifikasi WhatsApp")
total_frames = int(df_f["frame_count"].sum())
total_events = len(df_f)
total_msgs = int(df_f["wa_sent"].fillna(False).sum()) * 2  # start + summary
ratio = (total_msgs / total_frames * 100) if total_frames else 0
reduction = (1 - total_msgs / total_frames) * 100 if total_frames else 0
m1, m2, m3, m4 = st.columns(4)
m1.metric("Frame deteksi (akumulasi)", f"{total_frames:,}")
m2.metric("Event terdeteksi", f"{total_events:,}")
m3.metric("Pesan WA dikirim", f"{total_msgs:,}")
m4.metric("Reduksi vs 1-msg-per-frame", f"{reduction:.1f}%")
st.caption(
    "Tanpa event-debouncing, sistem akan mengirim 1 notifikasi per frame "
    "deteksi. Dengan event-debouncing, hanya 2 pesan (start + summary) "
    "per event."
)

# ---------------- Time-series ----------------
left, right = st.columns(2)
with left:
    st.subheader("Event per Jam")
    if len(df_f):
        hourly = (
            df_f.set_index("started_at_dt")
            .assign(n=1)
            .resample("1h")["n"].sum()
        )
        st.line_chart(hourly)
    else:
        st.info("Tidak ada data sesuai filter.")

with right:
    st.subheader("Distribusi Kelas Peak")
    if len(df_f):
        st.bar_chart(df_f["peak_class"].value_counts())
    else:
        st.info("Tidak ada data sesuai filter.")

# ---------------- Confidence & durasi ----------------
left2, right2 = st.columns(2)
with left2:
    st.subheader("Distribusi Peak Confidence")
    if len(df_f):
        bins = pd.cut(
            df_f["peak_confidence"],
            bins=[i / 10 for i in range(0, 11)],
            include_lowest=True,
        )
        counts = bins.value_counts().sort_index()
        counts.index = counts.index.astype(str)
        counts.index.name = "Peak confidence"
        st.bar_chart(counts)

with right2:
    st.subheader("Distribusi Durasi Event")
    if len(df_f):
        bins = pd.cut(
            df_f["duration_sec"],
            bins=[0, 1, 3, 5, 10, 30, 60, 120, 300, 99999],
            include_lowest=True,
        )
        counts = bins.value_counts().sort_index()
        counts.index = counts.index.astype(str)
        counts.index.name = "Durasi (detik)"
        st.bar_chart(counts)

# ---------------- Tabel & snapshot ----------------
st.subheader(f"Daftar Event ({len(df_f)})")
top_n = st.slider("Tampilkan event terbaru:", 5, 200, 50, step=5)
df_sorted = df_f.sort_values("started_at", ascending=False).head(top_n)

for _, row in df_sorted.iterrows():
    eid = row["event_id"]
    level_emoji = "🚨" if row["level"] == "confirm" else "⚠️"
    fp_state = row.get("is_false_positive")
    if fp_state is True:
        fp_tag = " · ❌ FALSE POSITIVE"
    elif fp_state is False:
        fp_tag = " · ✅ TRUE POSITIVE"
    else:
        fp_tag = ""

    title = (
        f"{level_emoji} {fmt_ts(row['started_at'])} · "
        f"{row['level']} · {row['peak_class']} "
        f"({float(row['peak_confidence']):.0%}) · "
        f"{float(row['duration_sec']):.1f}s · "
        f"{int(row['frame_count'])} frames{fp_tag}"
    )
    with st.expander(title):
        cc1, cc2 = st.columns([1, 2])
        snap = row.get("snapshot_path")
        snap_p = Path(snap) if snap else None
        if snap_p and snap_p.exists():
            cc1.image(str(snap_p), use_container_width=True)
        else:
            cc1.warning(f"Snapshot tidak ditemukan: `{snap}`")

        cc2.markdown(
            f"""
            - **Event ID:** `{eid}`
            - **Mulai:** {fmt_ts(row['started_at'])}
            - **Selesai:** {fmt_ts(row['ended_at'])}
            - **Durasi:** {float(row['duration_sec']):.2f} detik
            - **Jumlah frame deteksi:** {int(row['frame_count'])}
            - **Peak kelas:** {row['peak_class']} ({float(row['peak_confidence']):.2%})
            - **Breakdown kelas:** `{row.get('class_breakdown', {})}`
            - **Lokasi kamera:** {row.get('camera_location') or '-'}
            - **WA terkirim:** {'Ya' if row.get('wa_sent') else 'Tidak'}
            - **Label saat ini:** {fp_state if fp_state is not None else 'Belum dilabel'}
            """
        )
        b1, b2, b3 = cc2.columns(3)
        if b1.button("❌ False Positive", key=f"fp_{eid}"):
            write_label(eid, True)
            st.rerun()
        if b2.button("✅ True Positive", key=f"tp_{eid}"):
            write_label(eid, False)
            st.rerun()
        if b3.button("↺ Reset label", key=f"clr_{eid}"):
            write_label(eid, None)
            st.rerun()

# ---------------- Export ----------------
st.subheader("Ekspor Data")
csv = df_sorted.drop(columns=["started_at_dt", "ended_at_dt"], errors="ignore")
st.download_button(
    "📥 Download CSV (event yang difilter)",
    csv.to_csv(index=False).encode("utf-8"),
    file_name="events_filtered.csv",
    mime="text/csv",
)
