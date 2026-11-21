import gc
import json
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import ARRAY, REAL
from d23_lop4_nhom16.config import get_db_url, POSTGRES_DIR

DATABASE_URL = get_db_url()
engine = create_engine(DATABASE_URL)

def parse_vector_fast(x):
    if isinstance(x, str):
        x = x.strip().strip('"').strip("'")
        if not x.startswith("["):
            x = "[" + x
        if not x.endswith("]"):
            x = x + "]"
        return json.loads(x)
    return x


def import_csv_in_chunks(csv_path: Path, table_name: str, vector_cols=(), dtypes=None, chunk_rows=100):
    print(f"--> Insert '{table_name}' từ {csv_path.name} (chunk={chunk_rows})...")
    total_imported = 0
    first_chunk = True

    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):

        for col in vector_cols:
            if col in chunk.columns:
                chunk[col] = chunk[col].apply(parse_vector_fast)

        with engine.begin() as conn:
            chunk.to_sql(
                table_name,
                con=conn,
                if_exists="append",
                index=False,
                dtype=dtypes,
            )

        total_imported += len(chunk)
        print(f"   Inserted {total_imported} rows...", end="\r")
        gc.collect()

    print(f"\n   DONE: Table '{table_name}': {total_imported} rows.")


def run_import():
    print("=== STARTING INSERT DATA INTO POSTGRESQL ===")

    # 1. Metadata -> videos
    print("1/5 Importing metadata -> videos...")
    df_meta = pd.read_csv(POSTGRES_DIR / "metadata.csv")
    with engine.begin() as conn:
        df_meta.to_sql("videos", con=conn, if_exists="append", index=False, chunksize=500, method="multi")
    print(f"Imported {len(df_meta)} videos\n")

    # 2. Keyframes -> keyframes
    print("2/5 Importing keyframes -> keyframes...")
    df_kf = pd.read_csv(POSTGRES_DIR / "keyframes.csv")
    with engine.begin() as conn:
        df_kf.to_sql("keyframes", con=conn, if_exists="append", index=False, chunksize=500, method="multi")
    print(f"Imported {len(df_kf)} keyframes\n")

    # 3. Keyframe features (file lớn 860MB - nạp theo từng chunk 100 dòng)
    print("3/5 Importing keyframe_features -> keyframe_features...")
    import_csv_in_chunks(
        csv_path=POSTGRES_DIR / "keyframe_features.csv",
        table_name="keyframe_features",
        vector_cols=["hsv_vector", "glcm_vector", "hog_vector"],
        dtypes={
            "hsv_vector": ARRAY(REAL),
            "glcm_vector": ARRAY(REAL),
            "hog_vector": ARRAY(REAL),
        },
        chunk_rows=100,  # Giữ RAM luôn dưới 150MB
    )
    print()

    # 4. Flow features -> flow_features
    print("4/5 Importing flow_features -> flow_features...")
    df_flow = pd.read_csv(POSTGRES_DIR / "flow_features.csv")
    df_flow["flow_vector"] = df_flow["flow_vector"].apply(parse_vector_fast)
    with engine.begin() as conn:
        df_flow.to_sql("flow_features", con=conn, if_exists="append", index=False, dtype={"flow_vector": ARRAY(REAL)})
    print(f"Imported {len(df_flow)} flow features\n")

    # 5. MHI features -> mhi_features (nạp theo chunk 200 dòng)
    print("5/5 Importing mhi_features -> mhi_features...")
    import_csv_in_chunks(
        csv_path=POSTGRES_DIR / "mhi_features.csv",
        table_name="mhi_features",
        vector_cols=["mhi_vector"],
        dtypes={"mhi_vector": ARRAY(REAL)},
        chunk_rows=200,
    )

    print("\n DONE: Tất cả dữ liệu đã được nạp thành công vào PostgreSQL!")


if __name__ == "__main__":
    run_import()