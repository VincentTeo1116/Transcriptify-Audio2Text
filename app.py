import os
import json
import wave
import subprocess
import tempfile
import string
import re
import time
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from vosk import Model, KaldiRecognizer
import nltk
from google import genai

app = Flask(__name__)
CORS(app)

# initialization/configuration
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH","./vosk-model-small-en-us-0.15")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
CHUNK_SIZE = 3000
MAX_FILE_SIZE_MB = 200
MAX_DURATION_SECONDS = 900

# download NTLK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# helper functions
def get_audio_duration(file_path):
    # get duration in secs using ffprobe
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return None

def convert_to_wav(input_path, output_path, target_sr=16000, channels=1):
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", str(target_sr),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        "-y", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def clean_word(w):
    return w.translate(str.maketrans('','',string.punctuation)).lower()

def chunk_text(text, max_chars):
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    for w in words: 
        if current_len + len(w) + 1 > max_chars and current: 
            chunks.append(' '.join(current))
            current = [w]
            current_len = len(w) + 1
        else:
            current.append(w)
            current_len += len(w) + 1
    if current:
        chunks.append(' '.join(current))
    return chunks

def find_best_match(sent_words, raw_words, start_idx, window=15):
    best_score = 0
    best_start = start_idx
    best_end = start_idx
    search_start = max(0, start_idx - window)
    search_end = min(len(raw_words) - len(sent_words), start_idx + window)
    if search_end < search_start:
        return None
    for i in range(search_start, search_end + 1):
        raw_slice = raw_words[i:i + len(sent_words)]
        if not raw_slice:
            continue
        sm = SequenceMatcher(None, sent_words, raw_slice)
        score = sm.ratio()
        if score > best_score:
            best_score = score
            best_start = i
            best_end = i + len(sent_words) - 1
    if best_score > 0.6:
        return (best_start, best_end)
    return None

def correct_text_with_gemini(text_chunk, max_retries=5):
    prompt = """Fix the punctuation, capitalization, and grammar of the following speech-to-text transcript.

CRITICAL:
- DO NOT change, replace, add, or remove any word.
- ONLY add punctuation marks (commas, periods, question marks, etc.) and change capitalization (e.g., 'i' -> 'I').
- DO NOT correct spelling or word forms - just add punctuation and fix capitalization.
- Capitalize proper nouns like 'YYC', 'YYC Advisors' when they appear.
- Output ONLY the corrected text, nothing else.

Transcript:
""" + text_chunk

    client = genai.Client(api_key=GEMINI_API_KEY)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={'temperature': 0.0}
            )
            if response.text is None or response.text.strip() == "":
                if attempt == max_retries - 1:
                    return text_chunk
                time.sleep(2 ** attempt)
                continue
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                wait = 2 ** attempt
                print(f"Gemini busy, retry {attempt+1}/{max_retries} in {wait}s")
                time.sleep(wait)
            else:
                print(f"Gemini error: {e}")
                return text_chunk
    return text_chunk

# transcription func
def transcribe_audio(file_path):
    # Convert to WAV if needed
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        convert_to_wav(file_path, wav_path)
        wf = wave.open(wav_path, "rb")
    except Exception as e:
        raise RuntimeError(f"Audio conversion/opening failed: {e}")

    # load Vosk model
    if not os.path.exists(VOSK_MODEL_PATH):
        raise RuntimeError("Vosk model not found")
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    # transcribe
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    wf.close()
    os.unlink(wav_path)

    # extract words with timestamps
    word_timestamps = []
    full_text = ""
    for res in results:
        if 'result' in res:
            for info in res['result']:
                w = info['word']
                s = info['start']
                e = info['end']
                word_timestamps.append((s, e, w))
                full_text += w + " "
    full_text = full_text.strip()
    if not word_timestamps:
        raise RuntimeError("No words recognized")

    # chunk and correct with Gemini
    chunks = chunk_text(full_text, CHUNK_SIZE)
    corrected_chunks = []
    for chunk in chunks:
        corrected = correct_text_with_gemini(chunk)
        if corrected and corrected[-1] not in '.!?':
            corrected += '.'
        corrected_chunks.append(corrected)
        time.sleep(1)  # rate limit
    corrected_full_text = ' '.join(corrected_chunks)

    # post-process capitalization
    corrected_full_text = re.sub(r'\bi\b', 'I', corrected_full_text)
    corrected_full_text = corrected_full_text.replace('yyc', 'YYC')
    corrected_full_text = corrected_full_text.replace('yyc advisors', 'YYC Advisors')

    # sentence segmentation
    try:
        sentences = nltk.sent_tokenize(corrected_full_text, language='english')
        sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    except:
        sentences = [s.strip() for s in re.split(r'[.!?]', corrected_full_text) if s.strip()]
        sentences = [s.capitalize() + '.' for s in sentences]

    # align sentences to timestamps
    raw_words = [clean_word(w) for _, _, w in word_timestamps]
    sentence_indices = []
    idx = 0
    for sent in sentences:
        sent_clean = [clean_word(w) for w in sent.split()]
        sent_clean = [w for w in sent_clean if w]
        if not sent_clean:
            continue
        # exact match
        if (idx + len(sent_clean) <= len(raw_words) and
            raw_words[idx:idx+len(sent_clean)] == sent_clean):
            start_idx = idx
            end_idx = idx + len(sent_clean) - 1
            idx += len(sent_clean)
            sentence_indices.append((start_idx, end_idx))
            continue
        # fuzzy match
        match = find_best_match(sent_clean, raw_words, idx)
        if match:
            start_idx, end_idx = match
            idx = end_idx + 1
            sentence_indices.append((start_idx, end_idx))
        else:
            # If cannot align, assign approximate time from average word duration
            # (fallback: use the timestamp of the first word of the whole sentence)
            # For simplicity, we'll skip this sentence.
            print(f"Warning: Could not align: {sent[:50]}...")
            continue

    # build output
    output_sentences = []
    for i, (s_idx, e_idx) in enumerate(sentence_indices):
        start_time = word_timestamps[s_idx][0]
        end_time = word_timestamps[e_idx][1]
        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        output_sentences.append({
            "start": fmt(start_time),
            "end": fmt(end_time),
            "text": sentences[i] if i < len(sentences) else ""
        })
    return output_sentences

# flask route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        file_path = tmp.name

    # check duration
    duration = get_audio_duration(file_path)
    if duration is not None and duration > MAX_DURATION_SECONDS:
        # still process but frontend will show warning
        warning = f"File is {duration//60:.0f} minutes long – processing may take a while or time out."
    else:
        warning = None

    try:
        sentences = transcribe_audio(file_path)
        os.unlink(file_path)
        return jsonify({
            "success": True,
            "sentences": sentences,
            "warning": warning
        })
    except Exception as e:
        os.unlink(file_path)  # clean up
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)