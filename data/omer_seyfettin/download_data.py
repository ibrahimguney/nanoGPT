"""
Vikikaynak'tan Omer Seyfettin hikayelerini indirir.
"""
import os
import re
import time
import json
import urllib.request
import urllib.parse
import html

HIKAYELER = [
    "Forsa", "Pembe İncili Kaftan", "Kaşağı", "Yalnız Efe", "Diyet",
    "Kütük", "Yüksek Ökçeler", "Bahar ve Kelebekler", "Eleğimsağma",
    "Falaka", "Beyaz Lâle", "İlk Cinayet", "İlk Namaz", "Üç Nasihat",
    "Pireler", "Külah", "Ferman", "Topuz", "Velinimet", "Vire",
    "And", "Tos", "Mehdi", "Kurumuş Ağaçlar", "Tarih Ezeli Bir Tekerrürdür",
    "Teke Tek", "Teselli", "Hürriyet Bayrakları", "Havyar", "Kerâmet",
    "Kır Sineği", "Kıskançlık", "Korkunç Bir Ceza", "Kurbağa Duası",
    "Müjde", "Namus", "Nasıl Kurtarmış?", "Pembe Menekşe", "Perili Köşk",
    "Piç", "Rüşvet", "Terakki", "Yeni Bir Hediye", "Çakmak",
    "İffet", "İki Mebus", "İlkbahar", "Gizli Mâbed", "Primo Türk Çocuğu",
]


def get_wiki_text(title):
    api_url = "https://tr.wikisource.org/w/api.php"
    params = {
        'action': 'parse', 'page': title, 'format': 'json',
        'prop': 'text', 'formatversion': '2',
    }
    url = api_url + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        'User-Agent': 'nanoGPT-Training/1.0'
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    if 'error' in data:
        raise Exception(data['error'].get('info', 'Unknown error'))
    return data.get('parse', {}).get('text', '')


def html_to_text(html_content):
    html_content = re.sub(r'<table[^>]*>.*?</table>', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'<div class="reflist".*?</div>', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'<sup[^>]*>.*?</sup>', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'</p>', '\n\n', html_content)
    html_content = re.sub(r'<br\s*/?>', '\n', html_content)
    text = re.sub(r'<[^>]+>', '', html_content)
    text = html.unescape(text)
    text = re.sub(r'\[duzenle\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def download_story(title):
    try:
        html_content = get_wiki_text(title)
        return html_to_text(html_content)
    except Exception as e:
        print(f"  ! Hata: {title} -> {e}")
        return ""


def main():
    print("=" * 60)
    print("Vikikaynak'tan Omer Seyfettin hikayeleri indiriliyor")
    print("=" * 60)
    output_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    all_text = []
    success = 0
    for i, title in enumerate(HIKAYELER, 1):
        print(f"[{i}/{len(HIKAYELER)}] {title}...", end=' ', flush=True)
        text = download_story(title)
        if text and len(text) > 500:
            all_text.append(f"\n\n=== {title} ===\n\n{text}")
            print(f"OK ({len(text):,} karakter)")
            success += 1
        else:
            print("BOS")
        time.sleep(0.3)
    full_text = '\n'.join(all_text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print("\n" + "=" * 60)
    print(f"OK: {success}/{len(HIKAYELER)} hikaye indirildi")
    print(f"Toplam: {len(full_text):,} karakter")
    print(f"Dosya: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()