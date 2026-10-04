from flask import Flask, request, jsonify

from flask_cors import CORS

import cv2

import os

import numpy as np

import json

from PIL import Image, ImageChops, ImageOps

import easyocr

import re

import datetime


# =====================================================================
# KEY WORDS
# =====================================================================
KEYWORDS = [

    "republika",

    "srbija",

    "identitet",

    "ime",

    "prezime",

    "datum",

    "република",

    "србија",

    "име",

    "презиме",

    "датум",

    "лична"
]


app = Flask(__name__)

CORS(app)


# =====================================================================
# 🚀 AI OCR Engine Initialization
# =====================================================================
print("[+] Loading EasyOCR AI model...")

reader = easyocr.Reader(

    ['rs_cyrillic', 'en'],

    gpu=False
)


# =====================================================================
# 🔍 ANALYSIS FUNCTIONS
# =====================================================================
def scan_moire_score(image_path):

    img = cv2.imread(image_path, 0)

    if img is None:

        return 0

    img = cv2.GaussianBlur(img, (7, 7), 0)

    dft = np.fft.fft2(img)

    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

    rows, cols = img.shape

    crow, ccol = rows // 2, cols // 2

    magnitude_spectrum[

        crow - 30:crow + 30,

        ccol - 30:ccol + 30

    ] = 0

    return np.max(magnitude_spectrum)


def check_brightness(image_path):

    img = cv2.imread(image_path)

    if img is None:

        return False, 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    avg_brightness = np.mean(gray)

    return (avg_brightness <= 220), avg_brightness


# =====================================================================
# 📅 Extracts the birth date from the source
# =====================================================================
def extract_birth_date(text):

    if not text:

        return "N/A"

    # Normalization of some characters that are often misread by OCR.

    normalized = text.replace("O", "0")

    normalized = normalized.replace("o", "0")

    normalized = normalized.replace(",", ".")

    normalized = normalized.replace("/", ".")

    normalized = normalized.replace("-", ".")

    # DD.MM.YYYY

    match = re.search(

        r'\b(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])\.(19\d{2}|20\d{2})\b',

        normalized

    )

    if match:

        day = int(match.group(1))

        month = int(match.group(2))

        year = int(match.group(3))

        return f"{day:02d}.{month:02d}.{year:04d}"

    # DD MM YYYY

    match = re.search(

        r'\b(0?[1-9]|[12]\d|3[01])\s+(0?[1-9]|1[0-2])\s+(19\d{2}|20\d{2})\b',

        normalized

    )

    if match:

        day = int(match.group(1))

        month = int(match.group(2))

        year = int(match.group(3))

        return f"{day:02d}.{month:02d}.{year:04d}"

    # If OCR inserts a space between digits.

    compact = re.sub(r'\s+', '', normalized)

    match = re.search(

        r'\b(\d{2})\.(\d{2})\.(\d{4})\b',

        compact

    )

    if match:

        day = int(match.group(1))

        month = int(match.group(2))

        year = int(match.group(3))

        if 1 <= day <= 31 and 1 <= month <= 12:

            return f"{day:02d}.{month:02d}.{year:04d}"

    return "N/A"


# =====================================================================
# 🖼️ PHOTOSHOP / EDIT DETECTION
# =====================================================================
def check_photoshop(file_path):

    try:

        img = Image.open(file_path)

        info = str(img.info).lower()

        suspicious_sources = [

            "photoshop",

            "adobe",

            "gimp",

            "canva",

            "picsart"

        ]

        return any(s in info for s in suspicious_sources)

    except Exception:

        return False


# =====================================================================
# 🧪 ELA ANALYSIS
# =====================================================================
def perform_ela_analysis(image_path, quality=90):

    if not os.path.exists(image_path):

        return 0

    original = Image.open(image_path).convert("RGB")

    temp_file = "ela_temp.jpg"

    original.save(

        temp_file,

        "JPEG",

        quality=quality

    )

    temporary = Image.open(temp_file).convert("RGB")

    difference = ImageChops.difference(

        original,

        temporary

    )

    ela_score = np.mean(

        np.array(difference)

    )

    try:

        os.remove(temp_file)

    except Exception:

        pass

    return ela_score


# =====================================================================
# 🔎 SHARPNESS
# =====================================================================
def measure_sharpness(image_path):

    img = cv2.imread(image_path)

    if img is None:

        return 0

    gray = cv2.cvtColor(

        img,

        cv2.COLOR_BGR2GRAY

    )

    return cv2.Laplacian(

        gray,

        cv2.CV_64F

    ).var()


# =====================================================================
# 🧩 TEXTURE / NOISE
# =====================================================================
def save_noise_texture(image_path):

    img = cv2.imread(image_path, 0)

    if img is None:

        return

    laplacian = cv2.Laplacian(

        img,

        cv2.CV_64F

    )

    laplacian = np.uint8(

        np.absolute(laplacian)

    )

    isolated_noise = cv2.equalizeHist(

        laplacian

    )

    cv2.imwrite(

        "moire_tekstura_debug.jpg",

        isolated_noise

    )


# =====================================================================
# 👤 NAME
# =====================================================================
def extract_first_name(text):

    match = re.search(

        r'(Ime|Given\s?\w+)\W+([А-ЯA-Z\s]+?)(?=\sDatum|\sПол|\sPrezime|$)',

        text,

        re.IGNORECASE

    )

    if match:

        first_name = match.group(2).strip()

        return first_name

    return "N/A"


# =====================================================================
# 👤 LAST NAME
# =====================================================================
def extract_last_name(text):

    match = re.search(

        r'(Prezime|Surname)\W+([А-ЯA-Z\s]+?)(?=\sIme|\sGiven|$)',

        text,

        re.IGNORECASE

    )

    if match:

        last_name = match.group(2).strip()

        return last_name

    return "N/A"


# =====================================================================
# 🎂 AGE VERIFICATION
# =====================================================================
def is_adult(birth_date_string):

    if birth_date_string == "N/A":

        print("[DEBUG] Date not found -> age cannot be verified.")

        return False

    try:

        birth_date = datetime.datetime.strptime(

            birth_date_string,

            "%d.%m.%Y"

        ).date()

        today = datetime.date.today()

        age = today.year - birth_date.year

        # If your birthday hasn't happened yet this year,
        # The person is still a minor.

        if (

            (today.month, today.day)

            < (birth_date.month, birth_date.day)

        ):

            age -= 1

        print(

            f"[DEBUG] Date of birth: {birth_date_string} | "

            f"Age: {age}"

        )

        return age >= 18

    except Exception as e:

        print(

            f"[DEBUG] Error checking age: {e}"

        )

        return False


# =====================================================================
# 📍 FINDING FIELDS UNDER KEYWORD
# =====================================================================
def find_field_below(ocr_results, keyword):

    keyword_bbox = None

    for (bbox, text, prob) in ocr_results:

        if keyword.lower() in text.lower():

            keyword_bbox = bbox

            break

    if not keyword_bbox:

        return "N/A"

    keyword_y = keyword_bbox[0][1]

    keyword_x = keyword_bbox[0][0]

    candidates = []

    for (bbox, text, prob) in ocr_results:

        candidate_y = bbox[0][1]

        candidate_x = bbox[0][0]

        if (

            keyword_y

            < candidate_y

            < (keyword_y + 120)

        ):

            if abs(

                candidate_x - keyword_x

            ) < 200:

                if keyword.lower() not in text.lower():

                    if (

                        any(

                            "А" <= c <= "Я"

                            for c in text

                        )

                        and len(text) > 2

                    ):

                        candidates.append(

                            (

                                candidate_y,

                                text

                            )

                        )

                        print(

                            f"[DEBUG] Candidate for "

                            f"{keyword}: {text} "

                            f"at Y={candidate_y}"

                        )

    if candidates:

        candidates.sort(

            key=lambda x: x[0]

        )

        if keyword == "Име":

            return candidates[-1][1]

        else:

            return candidates[0][1]

    return "N/A"


# =====================================================================
# 🟢 GUILLOCHE LINES
# =====================================================================
def check_guilloche_lines(image_path):

    img = cv2.imread(image_path)

    if img is None:

        return False, 0

    gray = cv2.cvtColor(

        img,

        cv2.COLOR_BGR2GRAY

    )

    thresh = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV,

        11,

        2

    )

    pixel_structure = cv2.countNonZero(

        thresh

    )

    total_pixels = (

        img.shape[0] * img.shape[1]

    )

    density = (

        pixel_structure

        / total_pixels

    ) * 100

    return (

        density < 18.0,

        density

    )


# =====================================================================
# 🤖 EASY OCR + GEOMETRY
# =====================================================================
def check_geometry_and_text_easyocr(

    image_path

):

    img = cv2.imread(image_path)

    if img is None:

        return [], False, 0, False

    img_rgb = cv2.cvtColor(

        img,

        cv2.COLOR_BGR2RGB

    )

    try:

        results = reader.readtext(

            img_rgb

        )

        detected_text = ""

        static_heights = []

        data_heights = []

        screen_detected_from_text = False

        keywords = KEYWORDS

        screen_blacklist = [

            "paint",

            "photoshop",

            "gimp",

            "canva",

            "layers",

            "windows",

            "desktop",

            "file",

            "edit",

            "view",

            "image",

            "chrome",

            "edge"

        ]

        print(

            "\n--- WORDS DETECTED THROUGH AI ---"

        )

        for (

            bbox,

            text,

            prob

        ) in results:

            word = text.strip()

            if len(word) < 2 or prob < 0.51:

                continue

            detected_text += word + " "

            if any(

                keyword.lower()

                in word.lower()

                for keyword in keywords

            ):

                print(

                    f"[KEYWORD] {word} "

                    f"(CERTAINTY: {round(prob * 100)}%)"

                )

            else:

                print(

                    f"-> [{round(prob * 100)}%] "

                    f"READ: {word}"

                )

            if any(

                s in word.lower()

                for s in screen_blacklist

            ):

                screen_detected_from_text = True

            word_height = (

                bbox[2][1]

                - bbox[0][1]

            )

            if word.lower() in [

                "ime",

                "prezime",

                "datum",

                "republika",

                "srbija"

            ]:

                static_heights.append(

                    word_height

                )

            elif word.isupper() and len(word) > 3:

                data_heights.append(

                    word_height

                )

        print(

            f"[DEBUG] Total text passed "

            f"for verification: "

            f"{detected_text}"

        )

        avg_static = (

            np.mean(static_heights)

            if static_heights

            else 0

        )

        avg_data = (

            np.mean(data_heights)

            if data_heights

            else 0

        )

        geometry_ok = (

            1.4

            <= (avg_data / avg_static)

            <= 2.5

        ) if (

            avg_static > 0

            and avg_data > 0

        ) else False

        return (

            results,

            geometry_ok,

            avg_data,

            screen_detected_from_text

        )

    except Exception as e:

        print(

            f"[!] Error in OCR function: {e}"

        )

        return [], False, 0, False


# =====================================================================
# 📝 DATA EXTRACTION
# =====================================================================
def extract_data(text):

    text_lower = text.lower()

    def find_after(keyword):

        index = text_lower.find(

            keyword.lower()

        )

        if index != -1:

            remaining = text[

                index + len(keyword):

            ].strip()

            return (

                remaining.split()[0]

                if remaining

                else "N/A"

            )

        return "N/A"

    return {

        "ime": find_after("Given name"),

        "prezime": find_after("Surname"),

        "datum": find_after("Date of birth")

    }


# =====================================================================
# 🛣️ MAIN UPLOAD ROUTE
# =====================================================================
@app.route("/upload", methods=["POST"])
def upload():

    try:

        file = request.files["photo"]

        exif_raw = request.form.get(

            "exif",

            "{}"

        )

        exif_data = json.loads(

            exif_raw

        )

        file_path = "dolazna_slika.jpg"

        pil_img = Image.open(

            file.stream

        )

        new_width = int(

            pil_img.size[0] * 0.4

        )

        new_height = int(

            pil_img.size[1] * 0.4

        )

        pil_img = pil_img.resize(

            (

                new_width,

                new_height

            ),

            Image.Resampling.LANCZOS

        )

        pil_img.convert("RGB").save(

            file_path,

            "JPEG",

            quality=95

        )

        # =============================================================
        # 1. ANALYSIS
        # =============================================================

        ela_score = perform_ela_analysis(

            file_path

        )

        photoshop_detected = check_photoshop(

            file_path

        )

        sharpness_score = measure_sharpness(

            file_path

        )

        moire_value = scan_moire_score(

            file_path

        )

        guilloche_pass, guilloche_density = (

            check_guilloche_lines(

                file_path

            )

        )

        save_noise_texture(

            file_path

        )

        ocr_results, geometry_ok, _, screen_alarm = (

            check_geometry_and_text_easyocr(

                file_path

            )

        )

        # =============================================================
        # 2. TEXT RECONSTRUCTION
        # =============================================================

        text = " ".join(

            [

                result[1]

                for result in ocr_results

            ]

        )

        # =============================================================
        # 3. DATA EXTRACTION
        # =============================================================

        first_name = find_field_below(

            ocr_results,

            "Име"

        )

        last_name = find_field_below(

            ocr_results,

            "Презиме"

        )

        birth_date = extract_birth_date(

            text

        )

        print("\n=================================")

        print("[DEBUG] OCR TEXT:", text)

        print("[DEBUG] FIRST NAME:", first_name)

        print("[DEBUG] LAST NAME:", last_name)

        print("[DEBUG] DATE OF BIRTH:", birth_date)

        print("=================================")

        # =============================================================
        # 4. SCORING
        # =============================================================

        score = 0

        if photoshop_detected:

            score -= 5

        else:

            score += 6

        if ela_score < 30.0:

            score += 3

        if screen_alarm:

            score -= 6

        if sharpness_score > 2500:

            score -= 2

        if sharpness_score < 50:

            score -= 3

        if moire_value > 290:

            score -= 4

        if (

            not guilloche_pass

            and sharpness_score > 250

        ):

            score -= 4

        found_keywords = sum(

            1

            for keyword in KEYWORDS

            if keyword.lower() in text.lower()

        )

        if found_keywords == 0:

            score -= 5

        # =============================================================
        # 5. CONFIDENCE
        # =============================================================

        min_score = -15

        max_score = 20

        confidence = (

            (

                (score - min_score)

                / (max_score - min_score)

            )

            * 9

            + 1

        )

        confidence = round(

            max(

                1,

                min(10, confidence)

            ),

            1

        )

        # =============================================================
        # 6. IS PERSON AN ADULT OR MINOR
        # =============================================================

        adult = is_adult(

            birth_date

        )

        # =============================================================
        # DEBUG BEFORE FINAL SCORE
        # =============================================================

        print("\n=================================")

        print("[DEBUG] SCORE:", score)

        print("[DEBUG] CONFIDENCE:", confidence)

        print("[DEBUG] KEYWORDS FOUND:", found_keywords)

        print("[DEBUG] ADULT:", adult)

        print("[DEBUG] GUILLOCHE:", guilloche_pass)

        print("[DEBUG] ELA:", ela_score)

        print("[DEBUG] SHARPNESS:", sharpness_score)

        print("[DEBUG] MOIRE:", moire_value)

        print("[DEBUG] SCREEN ALARM:", screen_alarm)

        print("=================================")

        # =============================================================
        # 7. FINAL SCORE LOGIC
        # =============================================================

        if (

            adult

            and confidence >= 6

            and found_keywords >= 3

        ):

            status = "VERIFIKOVAN ✅"

        elif (

            confidence >= 5

            and found_keywords > 5

        ):

            status = (

                "POTREBNA RUČNA PROVERA ⚠️"

            )

        else:

            status = (

                "ODBIJEN - "

                "DETEKTOVAN LAŽNI IZVOR ❌"

            )

        print(

            "[DEBUG] FINAL VERDICT:",

            status

        )

        # =============================================================
        # 8. JSON INPUT
        # =============================================================

        if status != (

            "ODBIJEN - "

            "DETEKTOVAN LAŽNI IZVOR ❌"

        ):

            try:

                new_record = {

                    "vreme": datetime.datetime.now().strftime(

                        "%d.%m.%Y %H:%M:%S"

                    ),

                    "status": status,

                    "score": confidence,

                    "sadrzaj": text,

                    "datum_rodjenja": birth_date,

                    "ime": first_name,

                    "prezime": last_name

                }

                BASE_DIR = os.path.dirname(

                    os.path.abspath(__file__)

                )

                json_path = os.path.join(

                    BASE_DIR,

                    "podaci.json"

                )

                database = []

                if os.path.exists(

                    json_path

                ):

                    with open(

                        json_path,

                        "r",

                        encoding="utf-8"

                    ) as f:

                        try:

                            database = json.load(f)

                        except Exception:

                            database = []

                database.append(

                    new_record

                )

                with open(

                    json_path,

                    "w",

                    encoding="utf-8"

                ) as f:

                    json.dump(

                        database,

                        f,

                        indent=4,

                        ensure_ascii=False

                    )

                return jsonify({

                    "status": "success",

                    "presuda": status,

                    "score": confidence,

                    "datum_rodjenja": birth_date,

                    "ime": first_name,

                    "prezime": last_name,

                    "kljucne_reci": found_keywords,

                    "punoletan": adult

                })

            except Exception as e:

                print(

                    "[!] Error writing to database:",

                    repr(e)

                )

                return jsonify({

                    "status": "error",

                    "message": "Greška pri upisu u bazu"

                }), 500

        return jsonify({

            "status": "success",

            "presuda": status,

            "score": confidence,

            "datum_rodjenja": birth_date,

            "ime": first_name,

            "prezime": last_name,

            "kljucne_reci": found_keywords,

            "punoletan": adult

        })

    except Exception as e:

        print("\n" + "=" * 60)

        print("[❌] ERROR IN /upload")

        print(f"[❌] {repr(e)}")

        print("=" * 60 + "\n")

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# =====================================================================
# 🚀 SERVER START
# =====================================================================
if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5001,

        debug=False

    )