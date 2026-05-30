import requests
from bs4 import BeautifulSoup
import os
import time

# Each entry is (human-readable filename, URL)
PAGES = [
    # General
    ("Home",                                            "https://dioceseofkalookan.github.io/rcdok-website-new/"),
    ("About - History",                                 "https://dioceseofkalookan.github.io/rcdok-website-new/about-us/history/"),
    ("About - Our Bishop",                              "https://dioceseofkalookan.github.io/rcdok-website-new/about-us/our-bishop/"),
    ("Contact",                                         "https://dioceseofkalookan.github.io/rcdok-website-new/contact/"),
    ("Faith and Prayers",                               "https://dioceseofkalookan.github.io/rcdok-website-new/faith/faith/"),
    ("Events",                                          "https://dioceseofkalookan.github.io/rcdok-website-new/events/events/"),
    ("News",                                            "https://dioceseofkalookan.github.io/rcdok-website-new/news/"),
    ("Member Schools",                                  "https://dioceseofkalookan.github.io/rcdok-website-new/member-school/member-schools/"),
    ("Clergy - Diocesan",                               "https://dioceseofkalookan.github.io/rcdok-website-new/clergy/diocesan/"),
    ("Clergy - Religious",                              "https://dioceseofkalookan.github.io/rcdok-website-new/clergy/religious/"),
    ("Missions - Centers",                              "https://dioceseofkalookan.github.io/rcdok-website-new/missions/centers/"),
    ("Missions - Programs",                             "https://dioceseofkalookan.github.io/rcdok-website-new/missions/programs/"),
    ("Cemeteries",                                      "https://dioceseofkalookan.github.io/rcdok-website-new/cemeteries-columbaries-ossuaries/cemeteries/"),
    ("Columbaries",                                     "https://dioceseofkalookan.github.io/rcdok-website-new/cemeteries-columbaries-ossuaries/columbaries/"),

    # Vicariate of Our Lady of Grace
    ("Parish - Diocesan Shrine of Our Lady of Grace",   "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-grace/diocesan-shrine-of-our-lady-of-grace/"),
    ("Parish - Sagrada Familia",                        "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-grace/sagrada-familia-parish/"),
    ("Parish - San Jose (Agudo)",                       "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-grace/san-jose-parish-agudo/"),
    ("Parish - San Pancracio",                          "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-grace/san-pancracio-parish/"),
    ("Parish - Hearts of Jesus and Mary",               "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-grace/hearts-of-jesus-and-mary-parish/"),

    # Vicariate of Sacred Heart of Jesus
    ("Parish - Sacred Heart of Jesus (MBS)",            "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-heart/sacred-heart-of-jesus-parish-mbs/"),
    ("Parish - Birhen ng Lourdes",                      "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-heart/birhen-ng-lourdes-parish/"),
    ("Parish - St Gabriel the Archangel",               "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-heart/st-gabriel-the-archangel-parish/"),
    ("Parish - Sta Quiteria and St Francis of Assisi",  "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-heart/sta-quiteria-and-st-francis-of-assisi-parish/"),
    ("Parish - Sts Peter and John",                     "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-heart/sts-peter-and-john-parish/"),
    ("Parish - Our Lady of Lujan",                      "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-heart/our-lady-of-lujan-parish/"),

    # Vicariate of San Bartolome
    ("Parish - San Bartolome",                          "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-bartolome/san-bartolome-parish/"),
    ("Parish - Exaltation of the Holy Cross",           "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-bartolome/exaltation-of-the-holy-cross-parish/"),
    ("Parish - Diocesan Shrine of Immaculate Conception","https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-bartolome/diocesan-shrine-and-parish-of-immaculate-conception/"),
    ("Parish - San Antonio de Padua",                   "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-bartolome/san-antonio-de-padua-parish/"),
    ("Parish - Santa Cruz",                             "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-bartolome/santa-cruz-parish/"),
    ("Parish - Santo Rosario",                          "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-bartolome/santo-rosario-parish/"),

    # Vicariate of San Jose de Navotas
    ("Parish - San Ildefonso",                          "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-jose/san-ildefonso-parish/"),
    ("Parish - San Exequiel Moreno",                    "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-jose/san-exequiel-moreno-parish/"),
    ("Parish - San Roque de Navotas",                   "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-jose/san-roque-de-navotas-parish/"),
    ("Parish - Diocesan Shrine of San Jose de Navotas", "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-jose/diocesan-shrine-and-parish-of-san-jose-de-navotas/"),
    ("Parish - Sta Clare of Assisi",                    "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-jose/sta-clare-of-assisi-parish/"),
    ("Parish - San Lorenzo Ruiz and Companion Martyrs", "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-jose/san-lorenzo-ruiz-and-companion-martyrs-parish/"),
    ("Parish - Sto Niño de Pasion",                     "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-jose/sto-ni%C3%B1o-de-pasion-parish/"),
    ("Parish - Nuestra Señora de los Remedios",         "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-jose/nuestra-se%C3%B1ora-de-los-remedios-quasi-parish/"),

    # Vicariate of San Roque
    ("Parish - Immaculate Heart of Mary",               "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-roque/immaculate-heart-of-mary-parish/"),
    ("Parish - Mary Help of Christians",                "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-roque/mary-help-of-christians-parish/"),
    ("Parish - Sacred Heart of Jesus",                  "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-roque/sacred-heart-of-jesus-parish/"),
    ("Parish - St Joseph the Workman",                  "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-roque/st-joseph-the-workman-parish/"),
    ("Parish - San Roque Cathedral",                    "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-roque/san-roque-cathedral-parish/"),
    ("Parish - Holy Trinity",                           "https://dioceseofkalookan.github.io/rcdok-website-new/parishes/vicar-roque/holy-trinity-quasi-parish/"),
]

def scrape_page(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RCDoK-bot/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove nav, footer, scripts, styles — keep only content
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # Get the main text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive blank lines
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def name_to_filename(name):
    # Convert human-readable name to a safe filename
    # e.g. "Parish - Sto Niño de Pasion" → "Parish - Sto Niño de Pasion.txt"
    # Keep ñ and other accented characters — they're valid in filenames on Windows/Mac/Linux
    safe = name.replace("/", "-").replace("\\", "-").replace(":", "-").replace("*", "-")
    safe = safe.replace("?", "").replace('"', "").replace("<", "").replace(">", "").replace("|", "")
    return safe.strip() + ".txt"

def scrape_all():
    os.makedirs("knowledge_base", exist_ok=True)
    success, failed = 0, []

    for name, url in PAGES:
        print(f"Scraping: {name}")
        try:
            text = scrape_page(url)
            filename = name_to_filename(name)
            filepath = os.path.join("knowledge_base", filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"PAGE: {name}\n")
                f.write(f"SOURCE: {url}\n\n")
                f.write(text)
            print(f"  ✓ Saved → {filename} ({len(text)} chars)")
            success += 1
            time.sleep(0.5)  # be polite to the server
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append((name, url))

    print(f"\nDone! {success} pages saved to knowledge_base/")
    if failed:
        print(f"{len(failed)} failed:")
        for name, url in failed:
            print(f"  - {name}  ({url})")

scrape_all()