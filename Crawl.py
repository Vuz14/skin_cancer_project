import os
import csv
import time
import random
from botasaurus.browser import browser, Driver
from botasaurus.soupify import soupify
from global_land_mask import globe  # Thư viện cắt biển

# ==============================================================================
# 1. CẤU HÌNH (CONFIG) - CHẾ ĐỘ QUÁN CHAY
# ==============================================================================
OUTPUT_FILE = 'danang_quanchay_full.csv'  # Tên file mới
HISTORY_FILE = 'scanned_history_veg.txt'  # File lịch sử riêng cho quán chay
BATCH_SIZE = 5

# Tọa độ bao trọn Đà Nẵng (Từ Nam Ô -> FPT City -> Biển)
DANANG_BOUNDS = {
    "min_lat": 15.9200,
    "max_lat": 16.1700,
    "min_lng": 108.0600,
    "max_lng": 108.3200
}

GRID_STEP = 0.008

# --- TỪ KHÓA TÌM KIẾM (Đã mở rộng cho đồ chay) ---
SEARCH_KEYWORDS = [
    "Quán chay", "Cơm chay", "Bún chay", "Phở chay",
    "Lẩu chay", "Buffet chay", "Nhà hàng chay", "Vegan food",
    "Cơm chay"
]

# --- DANH MỤC HỢP LỆ ---
VALID_CATEGORIES = [
    "Vegetarian restaurant", "Vegan restaurant", "Buddhist temple",
    "Health food restaurant", "Macrobiotic restaurant",
    "Restaurant",

]

# --- TỪ KHÓA NHẬN DIỆN (Bắt buộc có nếu Category chung chung) ---
VEG_KEYWORDS = [
    "chay", "vegan", "vegetarian", "thực dưỡng", "nấm",
    "bồ đề", "an lạc", "thanh tịnh", "liên hoa", "hoa sen",
    "buddha"
]

# --- TỪ KHÓA LOẠI TRỪ (BLACKLIST) ---
# Đã tháo 'cơm/bún/phở', thêm các từ khóa thịt/cá/nhậu
BLACKLIST_KEYWORDS = [
    # Địa điểm không liên quan
    "spa", "nails", "massage", "hotel", "homestay", "resort", "gym", "yoga studio",
    "pharmacy", "thuốc", "bệnh viện", "store", "shop", "tạp hóa", "siêu thị",
    "bank", "atm", "school", "trường", "company", "công ty",

    # Đồ uống thuần túy (trừ khi tên có chữ chay)
    "cafe", "coffee", "cà phê", "trà sữa", "milk tea", "pub", "bar", "club",
    "karaoke", "internet", "gaming",

    # --- TỪ KHÓA MẶN (QUAN TRỌNG ĐỂ LỌC) ---
    "hải sản", "seafood", "ốc", "cua", "ghẹ", "tôm", "mực",
    "bê thui", "thịt chó", "cầy", "dê", "trâu", "lòng", "dồi",
    "tiết canh", "vịt quay", "heo quay", "nướng lu", "bbq",
    "steak", "bò né", "bò tơ", "bún đậu", "mắm tôm",
    "quán nhậu", "bia hơi", "mồi bén", "gà rán"
]


# ==============================================================================
# 2. HÀM QUẢN LÝ FILE
# ==============================================================================
def load_existing_ids():
    ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'place_id' in row: ids.add(row['place_id'])
            print(f"🔄 Đã load {len(ids)} quán cũ từ file CSV.")
        except:
            pass
    return ids


def load_history():
    points = set()
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    points.add(line.strip())
            print(f"🔄 Đã load {len(points)} điểm Grid đã hoàn thành từ lịch sử.")
        except:
            pass
    return points


def append_history(lat, lng):
    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{lat}_{lng}\n")


def save_batch(batch_data):
    if not batch_data: return
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        fieldnames = [
            "name", "address", "phone", "category", "rating",
            "reviews_count", "status", "opening_hours", "sample_reviews",
            "lat_scan", "lng_scan", "google_url", "place_id"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        writer.writerows(batch_data)
    print(f"    💾 [SAVED] Đã lưu {len(batch_data)} quán chay mới.")


# ==============================================================================
# 3. HÀM LOGIC & BỘ LỌC (IS_VALID_VEG)
# ==============================================================================
def is_valid_veg(name, category):
    name_lower = name.lower()
    cat_lower = category.lower()

    # 1. Loại trừ nếu dính từ khóa cấm (Mặn, Spa, Cafe...)
    for bad in BLACKLIST_KEYWORDS:
        if bad in name_lower or bad in cat_lower:
            return False

    # 2. Nếu Category khẳng định là quán chay -> LẤY NGAY
    priority_cats = ["vegetarian restaurant", "vegan restaurant", "nhà hàng ăn chay", "quán ăn chay"]
    if any(p_cat in cat_lower for p_cat in priority_cats):
        return True

    # 3. Nếu Category chung chung -> Check tên quán có từ khóa chay không
    # (Vd: "Cơm chay A Di Đà" -> Category: Restaurant -> Lọt vào đây check chữ "chay")
    if any(g in cat_lower for g in VALID_CATEGORIES):
        for kw in VEG_KEYWORDS:
            if kw in name_lower:
                return True

    return False


def get_place_id(url):
    try:
        return url.split("!1s")[1].split("!")[0] if "!1s" in url else url
    except:
        return url


def extract_hours(driver, soup):
    try:
        hours_btn = soup.select_one('[data-item-id="oh"] div[role="button"]')
        if hours_btn and hours_btn.get('aria-label'):
            return hours_btn.get('aria-label').replace("Hide open hours for the week", "").strip()
        status_div = soup.select_one('[data-item-id="oh"]')
        if status_div: return status_div.get_text(strip=True)
    except:
        pass
    return "N/A"


def extract_random_reviews(driver):
    try:
        review_btns = driver.select_all('button[role="tab"]')
        target_btn = None
        for btn in review_btns:
            txt = btn.text.lower() if btn.text else ""
            if "review" in txt or "đánh giá" in txt:
                target_btn = btn
                break
        if target_btn:
            target_btn.click()
            time.sleep(1.5)
            soup = soupify(driver)
            review_blocks = soup.select('div.jftiEf')
            temp_reviews = []
            for block in review_blocks:
                try:
                    content = block.select_one('.wiI7pd')
                    if content:
                        rating_el = block.select_one('span[role="img"]')
                        stars = rating_el['aria-label'] if rating_el else "[?]"
                        text = content.get_text(strip=True)
                        if len(text) > 10:
                            temp_reviews.append(f"{stars} {text}")
                except:
                    continue
            if temp_reviews:
                return " || ".join(random.sample(temp_reviews, min(len(temp_reviews), 5)))
    except:
        pass
    return ""


def parse_place_full(driver, url, lat_scan, lng_scan):
    try:
        driver.get(url)
        time.sleep(random.uniform(2.0, 3.0))
        soup = soupify(driver)

        try:
            name = soup.select_one('h1').get_text(strip=True)
        except:
            return None
        try:
            category = soup.select_one('button[jsaction*="category"]').get_text(strip=True)
        except:
            category = "Restaurant"

        # --- GỌI HÀM CHECK CHAY ---
        if not is_valid_veg(name, category): return None

        try:
            address = soup.select_one('[data-item-id="address"]').get_text(strip=True)
        except:
            address = ""
        try:
            phone = soup.select_one('[data-item-id*="phone"]').get_text(strip=True)
        except:
            phone = ""
        try:
            rating = soup.select_one('div.F7nice span[aria-hidden="true"]').get_text(strip=True)
        except:
            rating = ""
        try:
            reviews_count = soup.select_one('button[jsaction*="reviews"]').get_text(strip=True)
        except:
            reviews_count = "0"

        status = "Open"
        if "Permanently closed" in soup.get_text() or "Đã đóng cửa vĩnh viễn" in soup.get_text():
            status = "Closed"

        hours = extract_hours(driver, soup)
        sample_reviews = ""
        if status == "Open":
            sample_reviews = extract_random_reviews(driver)

        return {
            "name": name, "address": address, "phone": phone, "category": category,
            "rating": rating, "reviews_count": reviews_count, "status": status,
            "opening_hours": hours, "sample_reviews": sample_reviews,
            "lat_scan": lat_scan, "lng_scan": lng_scan,
            "google_url": url, "place_id": get_place_id(url)
        }
    except:
        return None


# ==============================================================================
# 4. ENGINE CHÍNH
# ==============================================================================
def generate_grid():
    points = []
    lat = DANANG_BOUNDS["min_lat"]
    while lat <= DANANG_BOUNDS["max_lat"]:
        lng = DANANG_BOUNDS["min_lng"]
        while lng <= DANANG_BOUNDS["max_lng"]:

            # --- CHECK BIỂN/ĐẢO ---
            # Chỉ lấy nếu là đất liền (Cần cài thư viện: pip install global-land-mask)
            if globe.is_land(lat, lng):
                points.append((lat, lng))

            lng += GRID_STEP
        lat += GRID_STEP
    return points


@browser(block_images=True, reuse_driver=True, close_on_crash=True, headless=True, output=None)
def run_final_crawler(driver: Driver, data):
    print(f"🚀 STARTING VEGAN CRAWLER (Batch Size: {BATCH_SIZE})...")

    crawled_ids = load_existing_ids()
    scanned_history = load_history()

    all_grid_points = generate_grid()

    # Lọc bỏ history
    grid_points = []
    for lat, lng in all_grid_points:
        key = f"{lat}_{lng}"
        if key not in scanned_history:
            grid_points.append((lat, lng))

    random.shuffle(grid_points)

    print(f"📊 Tổng Grid: {len(all_grid_points)}. Đã xong: {len(scanned_history)}. Cần chạy: {len(grid_points)}")

    current_batch = []

    for i, (lat, lng) in enumerate(grid_points):
        print(f"\n📍 Scanning Grid [{i + 1}/{len(grid_points)}]: {lat:.4f}, {lng:.4f}")
        found_count = 0

        for kw in SEARCH_KEYWORDS:
            try:
                driver.get(f"https://www.google.com/maps/search/{kw}/@{lat},{lng},14z")
                time.sleep(4)

                try:
                    if driver.select('div[role="feed"]'):
                        for _ in range(3):
                            driver.scroll('div[role="feed"]')
                            time.sleep(1.5)
                except:
                    pass

                elements = driver.select_all('a[href*="/maps/place/"]')
                for el in elements:
                    url = el.get_attribute('href')
                    if not url: continue
                    p_id = get_place_id(url)

                    if p_id in crawled_ids: continue
                    if any(item['place_id'] == p_id for item in current_batch): continue

                    data = parse_place_full(driver, url, lat, lng)

                    if data:
                        current_batch.append(data)
                        crawled_ids.add(p_id)
                        found_count += 1
                        print(f" ✅ LẤY: {data['name']}")
                    else:
                        crawled_ids.add(p_id)

                    if len(current_batch) >= BATCH_SIZE:
                        save_batch(current_batch)
                        current_batch.clear()

            except Exception as e:
                print(f"Err: {e}")

        append_history(lat, lng)
        if found_count == 0:
            print("   ⚠️ Grid này không tìm thấy quán chay mới.")

    if current_batch:
        save_batch(current_batch)

    print("\n🎉 HOÀN TẤT TOÀN BỘ!")


if __name__ == "__main__":
    run_final_crawler()