import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

# ======================
# ⚙️ Cấu hình
# ======================
START_URL = "https://www.hutech.edu.vn/"
OUTPUT_DIR = "data"
MAX_PAGES = 200  # Giới hạn trang để tránh quá tải
TIMEOUT = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# 🧠 Hàm tải file
# ======================
def download_file(url):
    filename = url.split("/")[-1].split("?")[0]
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".docx")):
        return False

    save_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(save_path):
        print(f"⚪ Bỏ qua (đã có): {filename}")
        return False

    try:
        print(f"⬇️  Đang tải: {filename}")
        r = requests.get(url, stream=True, timeout=TIMEOUT)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Tải xong: {filename}")
            return True
        else:
            print(f"❌ Lỗi {r.status_code} khi tải {filename}")
    except Exception as e:
        print(f"⚠️  Lỗi khi tải {filename}: {e}")
    return False


# ======================
# 🌐 Hàm crawl tự lan
# ======================
def crawl_website(start_url, max_pages=100):
    domain = urlparse(start_url).netloc
    visited = set()
    queue = deque([start_url])
    crawled_count = 0

    print(f"🚀 Bắt đầu crawl từ {start_url}\n")

    while queue and crawled_count < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            r = requests.get(url, timeout=TIMEOUT)
            if "text/html" not in r.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            # 🔍 Tải file trong trang
            for link in soup.find_all("a", href=True):
                file_url = urljoin(url, link["href"])
                if any(file_url.lower().endswith(ext) for ext in [".pdf", ".docx"]):
                    download_file(file_url)

                # 🔁 Thêm trang con vào hàng đợi nếu cùng domain
                elif urlparse(file_url).netloc == domain and file_url not in visited:
                    queue.append(file_url)

            crawled_count += 1
            print(f"🌍 Đã quét {crawled_count}/{max_pages} trang")

        except Exception as e:
            print(f"⚠️ Lỗi khi truy cập {url}: {e}")

    print(f"\n✅ Hoàn tất crawl {crawled_count} trang. Các file nằm trong thư mục '{OUTPUT_DIR}'.")


# ======================
# ▶️ Chạy chính
# ======================
if __name__ == "__main__":
    crawl_website(START_URL, MAX_PAGES)
