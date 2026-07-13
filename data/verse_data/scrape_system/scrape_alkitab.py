"""
Scraper untuk Alkitab Terjemahan Baru (TB) dari alkitab.mobi

Sumber data: http://alkitab.mobi/tb/{book_abbr}/{chapter}
Teknik: Sama seperti sonnylazuardi/alkitab-api (cheerio → BeautifulSoup)

Jalankan sekali untuk menghasilkan data/alkitab_tb.csv
"""

import requests
import csv
import time
import re
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

# === Daftar 66 Buku Alkitab (Singkatan alkitab.mobi + Nama Lengkap) ===
BOOKS = [
    # Perjanjian Lama (39 buku)
    ("Kej", "Kejadian", 50),
    ("Kel", "Keluaran", 40),
    ("Ima", "Imamat", 27),
    ("Bil", "Bilangan", 36),
    ("Ula", "Ulangan", 34),
    ("Yos", "Yosua", 24),
    ("Hak", "Hakim-Hakim", 21),
    ("Rut", "Rut", 4),
    ("1Sa", "1 Samuel", 31),
    ("2Sa", "2 Samuel", 24),
    ("1Ra", "1 Raja-Raja", 22),
    ("2Ra", "2 Raja-Raja", 25),
    ("1Ta", "1 Tawarikh", 29),
    ("2Ta", "2 Tawarikh", 36),
    ("Ezr", "Ezra", 10),
    ("Neh", "Nehemia", 13),
    ("Est", "Ester", 10),
    ("Ayb", "Ayub", 42),
    ("Mzm", "Mazmur", 150),
    ("Ams", "Amsal", 31),
    ("Pkh", "Pengkhotbah", 12),
    ("Kid", "Kidung Agung", 8),
    ("Yes", "Yesaya", 66),
    ("Yer", "Yeremia", 52),
    ("Rat", "Ratapan", 5),
    ("Yeh", "Yehezkiel", 48),
    ("Dan", "Daniel", 12),
    ("Hos", "Hosea", 14),
    ("Yoe", "Yoel", 3),
    ("Amo", "Amos", 9),
    ("Oba", "Obaja", 1),
    ("Yun", "Yunus", 4),
    ("Mik", "Mikha", 7),
    ("Nah", "Nahum", 3),
    ("Hab", "Habakuk", 3),
    ("Zef", "Zefanya", 3),
    ("Hag", "Hagai", 2),
    ("Zak", "Zakharia", 14),
    ("Mal", "Maleakhi", 4),
    # Perjanjian Baru (27 buku)
    ("Mat", "Matius", 28),
    ("Mrk", "Markus", 16),
    ("Luk", "Lukas", 24),
    ("Yoh", "Yohanes", 21),
    ("Kis", "Kisah Para Rasul", 28),
    ("Rom", "Roma", 16),
    ("1Ko", "1 Korintus", 16),
    ("2Ko", "2 Korintus", 13),
    ("Gal", "Galatia", 6),
    ("Efe", "Efesus", 6),
    ("Flp", "Filipi", 4),
    ("Kol", "Kolose", 4),
    ("1Te", "1 Tesalonika", 5),
    ("2Te", "2 Tesalonika", 3),
    ("1Ti", "1 Timotius", 6),
    ("2Ti", "2 Timotius", 4),
    ("Tit", "Titus", 3),
    ("Flm", "Filemon", 1),
    ("Ibr", "Ibrani", 13),
    ("Yak", "Yakobus", 5),
    ("1Pt", "1 Petrus", 5),
    ("2Pt", "2 Petrus", 3),
    ("1Yo", "1 Yohanes", 5),
    ("2Yo", "2 Yohanes", 1),
    ("3Yo", "3 Yohanes", 1),
    ("Yud", "Yudas", 1),
    ("Why", "Wahyu", 22),
]

BASE_URL = "http://alkitab.mobi/tb"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "alkitab_tb.csv")
DELAY_SECONDS = 0.5  # Rate limiting — jeda antar request


def scrape_chapter(book_abbr, chapter):
    """
    Scrape satu pasal dari alkitab.mobi dan kembalikan list ayat.
    
    Returns:
        list of dict: [{verse: int, text: str}, ...]
    """
    url = f"{BASE_URL}/{book_abbr}/{chapter}/"
    
    try:
        response = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Bible Research Bot - Academic Project)"
        })
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"\n  [ERROR] Gagal mengambil {url}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    verses = []
    
    # Setiap ayat ada di dalam <p> yang berisi <span class="reftext"> untuk nomor ayat
    # dan <span data-begin="..."> untuk konten ayat
    for p_tag in soup.find_all("p"):
        # Skip hidden, loading, error paragraphs
        if p_tag.get("hidden") == "hidden":
            continue
        if p_tag.get("class") and ("loading" in p_tag.get("class") or "error" in p_tag.get("class")):
            continue
        
        # Cek apakah ini title (section heading) — skip
        title_span = p_tag.find("span", class_="paragraphtitle")
        if title_span:
            continue
        
        # Cari nomor ayat
        ref_span = p_tag.find("span", class_="reftext")
        if not ref_span:
            continue
        
        # Ambil nomor ayat dari link di dalam reftext
        verse_link = ref_span.find("a")
        if verse_link:
            verse_num_text = verse_link.get_text(strip=True)
        else:
            verse_num_text = ref_span.get_text(strip=True)
        
        try:
            verse_num = int(verse_num_text)
        except (ValueError, TypeError):
            continue
        
        # Ambil teks ayat — hapus reftext dulu supaya nomor ayat tidak ikut
        # Clone p_tag agar tidak merusak soup asli
        p_copy = BeautifulSoup(str(p_tag), "html.parser").find("p")
        for ref in p_copy.find_all("span", class_="reftext"):
            ref.decompose()
        for title in p_copy.find_all("span", class_="paragraphtitle"):
            title.decompose()
        
        text = p_copy.get_text(strip=True)
        # Bersihkan whitespace berlebihan
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:
            verses.append({
                "verse": verse_num,
                "text": text
            })
    
    return verses


def run_scraper():
    """Jalankan proses scraping seluruh Alkitab TB."""
    
    # Hitung total pasal
    total_chapters = sum(chapters for _, _, chapters in BOOKS)
    print(f"Memulai scraping Alkitab Terjemahan Baru (TB)")
    print(f"Total: {len(BOOKS)} buku, {total_chapters} pasal")
    print(f"Sumber: {BASE_URL}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Jeda antar request: {DELAY_SECONDS}s")
    print("=" * 60)
    
    all_verses = []
    failed_chapters = []
    
    with tqdm(total=total_chapters, desc="Scraping", unit="pasal") as pbar:
        for book_abbr, book_name, num_chapters in BOOKS:
            for chapter in range(1, num_chapters + 1):
                pbar.set_postfix_str(f"{book_name} {chapter}")
                
                verses = scrape_chapter(book_abbr, chapter)
                
                if not verses:
                    failed_chapters.append(f"{book_name} {chapter}")
                
                for v in verses:
                    reference = f"{book_name} {chapter}:{v['verse']}"
                    all_verses.append({
                        "book_abbr": book_abbr,
                        "book_name": book_name,
                        "chapter": chapter,
                        "verse": v["verse"],
                        "text": v["text"],
                        "reference": reference
                    })
                
                pbar.update(1)
                time.sleep(DELAY_SECONDS)
    
    # Simpan ke CSV
    print(f"\nMenyimpan {len(all_verses)} ayat ke {OUTPUT_PATH}...")
    
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["book_abbr", "book_name", "chapter", "verse", "text", "reference"])
        writer.writeheader()
        writer.writerows(all_verses)
    
    print(f"Selesai! {len(all_verses)} ayat berhasil disimpan.")
    
    if failed_chapters:
        print(f"\n[WARNING] {len(failed_chapters)} pasal gagal di-scrape:")
        for fc in failed_chapters:
            print(f"  - {fc}")
    
    return all_verses


if __name__ == "__main__":
    run_scraper()
