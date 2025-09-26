import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---- Cấu hình đường dẫn ----
data_path = "output/cleaned_dataset.csv"  # <-- sửa nếu file nằm chỗ khác
out_fig_dir = Path("output/figures")
out_fig_dir.mkdir(parents=True, exist_ok=True)

# ---- Load và chuẩn hóa tên cột ----
df = pd.read_csv(data_path)

# chuẩn hóa tên cột: remove spaces, lowercase, replace '-' bằng '_' để dễ dùng
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

# ---- Lấy top 10 quốc gia theo tuổi thọ trung bình ----
# đảm bảo cột life_expectancy tồn tại
if "life_expectancy" not in df.columns:
    raise ValueError("Không tìm thấy cột 'life_expectancy' trong file. Kiểm tra lại tên cột.")

top10_countries = (
    df.groupby("country")["life_expectancy"]
      .mean()
      .nlargest(10)
      .index
      .tolist()
)

df_top10 = df[df["country"].isin(top10_countries)].copy()

# Đảm bảo cột year là numeric để vẽ
df_top10["year"] = pd.to_numeric(df_top10["year"], errors="coerce")

# Helper: tìm cột hiện có trong dataframe từ 1 danh sách tên khả dĩ
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: tìm theo chứa substring
    for cand in candidates:
        for col in df.columns:
            if cand in col:
                return col
    return None

# Danh sách các biến (mỗi mục là list tên khả dĩ) + nhãn tiếng Việt cho trục y và tiêu đề
plots = [
    (["life_expectancy"], "Tuổi thọ trung bình", "Xu hướng tuổi thọ trung bình (Top 10 quốc gia)"),
    (["adult_mortality"], "Tử vong người lớn (15-60 tuổi)", "Xu hướng tử vong ở người lớn (15–60 tuổi)"),
    (["alcohol"], "Mức tiêu thụ rượu (lit/người/năm)", "Xu hướng sử dụng rượu bia"),
    (["percentage_expenditure"], "Tỷ lệ GDP chi cho y tế (cột percentage_expenditure)", "Xu hướng tỷ lệ GDP chi cho y tế"),
    (["hepatitis_b"], "Tỷ lệ tiêm Viêm gan B (%)", "Xu hướng tiêm phòng Viêm gan B"),
    (["bmi"], "BMI trung bình", "Xu hướng chỉ số BMI"),
    (["under_five_deaths", "under-five_deaths", "under_five_death"], "Tử vong trẻ dưới 5 tuổi", "Xu hướng tỷ lệ trẻ tử vong dưới 5 tuổi"),
    (["total_expenditure"], "Tỷ lệ chi tiêu y tế trên tổng chi tiêu chính phủ (%) (total_expenditure)", "Xu hướng tỷ lệ chi tiêu y tế trên tổng chi tiêu chính phủ"),
    (["gdp"], "GDP bình quân đầu người", "Xu hướng GDP"),
    (["thinness_1_19_years", "thinness_1_19", "thinness_10_19_years"], "Tỷ lệ suy dinh dưỡng trẻ 10-19 (%)", "Xu hướng tỷ lệ suy dinh dưỡng trẻ em & thanh thiếu niên 10–19"),
    (["income_composition_of_resources"], "Chỉ số phát triển con người (0-1) - proxy", "Xu hướng Chỉ số phát triển con người (proxy)"),
    (["schooling"], "Bình quân số năm đi học", "Xu hướng Bình quân tuổi đến trường (số năm)")
]

# Cài đặt style seaborn
sns.set(style="whitegrid", context="talk", palette="tab10")

# Vẽ tuần tự từng biểu đồ
for candidates, ylabel, title in plots:
    col = find_col(df_top10, candidates)
    if col is None:
        print(f"[SKIP] Không tìm thấy cột phù hợp cho: {title} (các ứng viên: {candidates})")
        continue

    # chuyển thành numeric (nếu có chuỗi)
    df_top10[col] = pd.to_numeric(df_top10[col], errors="coerce")

    plt.figure(figsize=(12, 6))
    sns.lineplot(
    data=df_top10,
    x="year",
    y=col,
    hue="country",
    hue_order=top10_countries,
    marker="o",
    estimator=None,
    lw=1.5
    )
    plt.title(f"{title}")
    plt.ylabel(ylabel)
    plt.xlabel("Năm")
# Giới hạn trục X để hiển thị đủ 2000–2015
    plt.xlim(2000, 2015)
    plt.xticks(range(2000, 2016))   # bắt buộc hiện đủ nhãn năm
# ---- Legend chỉnh nhỏ + đặt ngoài hình ----
    plt.legend(
    title="Quốc gia",
    bbox_to_anchor=(1.05, 1),   # đặt legend ra ngoài bên phải
    loc="upper left",           # căn góc trên bên trái của legend vào điểm (1.05,1)
    fontsize="small",           # cỡ chữ nhỏ hơn
    title_fontsize="small"      # cỡ chữ tiêu đề nhỏ
    )
    plt.tight_layout()
    
    # Lưu file hình
    safe_col_name = col.replace("/", "_").replace(" ", "_")
    fname = out_fig_dir / f"trend_top10_{safe_col_name}.png"
    plt.savefig(fname, dpi=150)
    print(f"[SAVED] {fname}")
    plt.show()
