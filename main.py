# main.py - Complete Simplified Backend with Words Per Subtitle Control
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import openai 
import tempfile
import os
import uuid
import subprocess
import re
import json
import requests
import shutil

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

app = Flask(__name__)
CORS(app)

class Segment:
    def __init__(self, text, start, end):
        self.start = start
        self.end = end
        self.text = text

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',')

def generate_srt(segments):
    """Generate SRT subtitle file content"""
    srt = ""
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip().replace('-->', '→')
        srt += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt

def get_video_dimensions(video_path: str) -> tuple[int, int] | None:
    """Get video dimensions using ffprobe"""
    if not shutil.which("ffprobe"):
        app.logger.warning("ffprobe not found. Using default dimensions.")
        return None

    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split('x'))
        return width, height
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Error getting video dimensions: {e.stderr}")
        return None
    except Exception as e:
        app.logger.error(f"Exception in get_video_dimensions: {e}")
        return None

def create_subtitle_segments(words, keywords, words_per_subtitle=5):
    """Create subtitle segments by grouping words"""
    if not words:
        return []
    
    # Clamp words per subtitle to reasonable range
    words_per_subtitle = max(1, min(words_per_subtitle, 10))
    segments = []
    
    app.logger.info(f"Creating subtitles with {words_per_subtitle} words per segment")
    
    for i in range(0, len(words), words_per_subtitle):
        group = words[i:i+words_per_subtitle]
        if not group:
            continue
            
        # Combine text from all words in the group
        text = ' '.join([w.get('text', '') for w in group])
        text = highlight_keywords(text, keywords)
        
        # Use timing from first and last word
        start = float(group[0].get('start', 0))
        end = float(group[-1].get('end', 0))
        
        # Ensure minimum duration of 0.5 seconds
        if end - start < 0.5:
            end = start + 0.5
            
        segments.append(Segment(text, start, end))
    
    app.logger.info(f"Created {len(segments)} subtitle segments")
    return segments

def highlight_keywords(text, keywords):
    """Highlight keywords in subtitles with yellow color"""
    for kw in keywords:
        pattern = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
        text = pattern.sub(lambda m: f"<font color=\"yellow\">{m.group(0)}</font>", text)
    return text

def download_image(url, dest_path):
    """Download image from URL to local path"""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        app.logger.error(f"Failed to download {url}: {e}")
    return False

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio/video file using OpenAI Whisper"""
    file = request.files['file']
    language = request.form.get('language', None)

    safe_filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    temp_input_path = os.path.join(tempfile.gettempdir(), f"{unique_id}_{safe_filename}")
    file.save(temp_input_path)

    try:
        with open(temp_input_path, "rb") as audio_file:
            transcript_response = openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                language=language,
                timestamp_granularities=["segment", "word"]
            )
        
        raw_segments = transcript_response.segments
        full_text = transcript_response.text
        raw_words = getattr(transcript_response, "words", None)

        app.logger.info(f"Transcription completed: {len(raw_words or [])} words, {len(raw_segments or [])} segments")

    except Exception as e:
        app.logger.error(f"Error in transcription: {e}")
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return jsonify({"error": "Transcription failed", "details": str(e)}), 500

    # Clean up temp file
    if os.path.exists(temp_input_path):
        os.remove(temp_input_path)

    return jsonify({
        "text": full_text,
        "segments": [
            {"text": s.text, "start": s.start, "end": s.end}
            for s in raw_segments
        ] if raw_segments else [],
        "words": [
            {"text": w.word, "start": w.start, "end": w.end} 
            for w in raw_words
        ] if raw_words else [],
        "language": getattr(transcript_response, "language", None)
    })

@app.route('/extract-keywords', methods=['POST'])
def extract_keywords():
    """Extract keywords from transcript using OpenAI"""
    data = request.json
    transcript = data.get("transcript", "")

    if not transcript:
        return jsonify({"error": "Transcript required"}), 400

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Extract the most important keywords from this transcript. Focus on nouns, proper nouns, and key concepts. Return only a comma-separated list of single words or short phrases that would make good search terms for images."
                },
                {"role": "user", "content": transcript}
            ]
        )
        keywords_text = response.choices[0].message.content
        keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        
        app.logger.info(f"Extracted {len(keywords)} keywords: {keywords}")
        return jsonify({"keywords": keywords})
        
    except Exception as e:
        app.logger.error(f"Error extracting keywords: {e}")
        return jsonify({"error": "Keyword extraction failed", "details": str(e)}), 500

@app.route('/broll-images', methods=['POST'])
def broll_images():
    """Fetch B-roll images from Unsplash for keywords"""
    data = request.json
    keywords = data.get("keywords", [])
    
    if not keywords:
        return jsonify({"error": "No keywords provided"}), 400

    if not UNSPLASH_ACCESS_KEY:
        return jsonify({"error": "Unsplash API key not configured"}), 400

    images = {}
    failed_keywords = []
    access_token_invalid = False

    app.logger.info(f"Fetching B-roll images for {len(keywords)} keywords")

    for kw in keywords:
        url = f"https://api.unsplash.com/photos/random?query={kw}&client_id={UNSPLASH_ACCESS_KEY}&orientation=landscape"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 401:
                access_token_invalid = True
                images[kw] = None
                failed_keywords.append(kw)
            elif resp.status_code == 200:
                img_data = resp.json()
                if 'urls' in img_data and 'regular' in img_data['urls']:
                    images[kw] = img_data['urls']['regular']
                    app.logger.info(f"Found image for keyword: {kw}")
                else:
                    images[kw] = None
                    failed_keywords.append(kw)
            else:
                app.logger.error(f"Unsplash API error for '{kw}': {resp.status_code}")
                images[kw] = None
                failed_keywords.append(kw)
        except requests.exceptions.Timeout:
            app.logger.warning(f"Timeout fetching image for '{kw}'")
            images[kw] = None
            failed_keywords.append(kw)
        except Exception as e:
            app.logger.error(f"Exception fetching image for '{kw}': {e}")
            images[kw] = None
            failed_keywords.append(kw)

    result = {"images": images, "errors": failed_keywords}
    if access_token_invalid:
        result["access_token_invalid"] = True
    
    app.logger.info(f"B-roll fetch completed: {len([v for v in images.values() if v])} successful, {len(failed_keywords)} failed")
    return jsonify(result)

@app.route('/generate-video', methods=['POST'])
def generate_video():
    """Generate video with subtitles and B-roll overlays"""
    file = request.files['file']
    transcript_text = request.form.get('transcript', "")
    words_json = request.form.get('words', None)
    keywords_json = request.form.get('keywords', None)
    broll_images_json = request.form.get('broll_images', None)
    words_per_subtitle = int(request.form.get('words_per_subtitle', 5))

    # Parse JSON data
    keywords = json.loads(keywords_json) if keywords_json else []
    broll_images = json.loads(broll_images_json) if broll_images_json else {}
    words = json.loads(words_json) if words_json else []

    app.logger.info(f"=== VIDEO GENERATION STARTED ===")
    app.logger.info(f"Words: {len(words)}, Keywords: {len(keywords)}, Words per subtitle: {words_per_subtitle}")
    app.logger.info(f"B-roll images: {len([v for v in broll_images.values() if v])}")

    if not transcript_text:
        return jsonify({"error": "Transcript required"}), 400

    # Save uploaded file
    safe_filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    temp_input_path = os.path.join(tempfile.gettempdir(), f"{unique_id}_{safe_filename}")
    file.save(temp_input_path)
    temp_files = [temp_input_path]

    try:
        # Create subtitle segments
        subtitle_segments = create_subtitle_segments(words, keywords, words_per_subtitle)
        
        if not subtitle_segments:
            # Fallback: create segments from text
            app.logger.warning("No word-level data, creating basic segments")
            sentences = re.split(r'(?<=[.!?])\s+', transcript_text.strip())
            current_time = 0.0
            for sentence in sentences:
                if sentence.strip():
                    text = highlight_keywords(sentence.strip(), keywords)
                    subtitle_segments.append(Segment(text, current_time, current_time + 3.0))
                    current_time += 3.0

        # Prevent overlapping subtitles
        for i in range(len(subtitle_segments) - 1):
            current = subtitle_segments[i]
            next_seg = subtitle_segments[i + 1]
            if current.end >= next_seg.start:
                current.end = next_seg.start - 0.1
                # Ensure minimum duration
                if current.end - current.start < 0.5:
                    current.end = current.start + 0.5

        # Generate SRT file
        temp_srt_path = temp_input_path + ".srt"
        output_video_path = temp_input_path + "_final.mp4"
        temp_files.extend([temp_srt_path, output_video_path])

        srt_content = generate_srt(subtitle_segments)
        with open(temp_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        app.logger.info(f"Generated SRT with {len(subtitle_segments)} segments")

        # Get video dimensions
        dimensions = get_video_dimensions(temp_input_path)
        target_width, target_height = dimensions if dimensions else (1280, 720)
        app.logger.info(f"Video dimensions: {target_width}x{target_height}")

        # Download B-roll images
        downloaded_broll = {}
        for kw, url in broll_images.items():
            if url:
                img_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{secure_filename(kw)}.jpg")
                if download_image(url, img_path):
                    downloaded_broll[kw] = img_path
                    temp_files.append(img_path)
                    app.logger.info(f"Downloaded B-roll image for: {kw}")

        app.logger.info(f"Downloaded {len(downloaded_broll)} B-roll images")

        # Build FFmpeg command
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_input_path]
        
        # Add B-roll images as inputs
        input_idx = 1
        broll_idx_map = {}
        for kw, img_path in downloaded_broll.items():
            ffmpeg_cmd.extend(["-i", img_path])
            broll_idx_map[kw] = input_idx
            input_idx += 1

        # Create filter complex for B-roll overlays
        filter_parts = []
        video_stream = "0:v"
        overlay_count = 0

        # Scale main video
        filter_parts.append(f"[{video_stream}]scale={target_width}:{target_height}[scaled_main]")
        video_stream = "scaled_main"

        # Add B-roll overlays based on word timing
        if words and downloaded_broll:
            app.logger.info("Adding B-roll overlays with word-level timing")
            
            overlays_added = 0
            for word in words:
                word_text = word.get('text', '').strip().lower()
                word_text = re.sub(r'[^\w\s]', '', word_text)  # Remove punctuation
                word_start = float(word.get('start', 0))
                word_end = float(word.get('end', 0))
                
                # Check if word matches any keyword
                for kw in keywords:
                    kw_clean = re.sub(r'[^\w\s]', '', kw.lower())
                    
                    if word_text == kw_clean and kw in downloaded_broll:
                        img_idx = broll_idx_map[kw]
                        overlay_label = f"overlay_{overlay_count}"
                        
                        # Add B-roll overlay that appears when keyword is spoken
                        filter_parts.append(
                            f"[{video_stream}][{img_idx}:v]"
                            f"overlay=x=(W-w)/2:y=(H-h)/2:enable='between(t,{word_start},{word_end})'"
                            f"[{overlay_label}]"
                        )
                        video_stream = overlay_label
                        overlay_count += 1
                        overlays_added += 1
                        app.logger.info(f"Added B-roll for '{kw}' at {word_start:.1f}-{word_end:.1f}s")
                        break
            
            app.logger.info(f"Total B-roll overlays added: {overlays_added}")
        
        elif downloaded_broll:
            # Fallback: use segment-based timing
            app.logger.warning("Using segment-based B-roll timing")
            for segment in subtitle_segments:
                segment_text = segment.text.lower()
                # Remove HTML tags for matching
                clean_text = re.sub(r'<[^>]+>', '', segment_text)
                
                for kw in keywords:
                    if kw.lower() in clean_text and kw in downloaded_broll:
                        img_idx = broll_idx_map[kw]
                        overlay_label = f"overlay_{overlay_count}"
                        
                        filter_parts.append(
                            f"[{video_stream}][{img_idx}:v]"
                            f"overlay=x=(W-w)/2:y=(H-h)/2:enable='between(t,{segment.start},{segment.end})'"
                            f"[{overlay_label}]"
                        )
                        video_stream = overlay_label
                        overlay_count += 1
                        app.logger.info(f"Added segment-based B-roll for '{kw}'")
                        break

        # Add subtitles
        srt_path_escaped = temp_srt_path.replace('\\', '/').replace(':', '\\:')
        filter_parts.append(f"[{video_stream}]subtitles='{srt_path_escaped}'[final]")

        # Complete FFmpeg command
        filter_complex = ";".join(filter_parts)
        ffmpeg_cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[final]",
            "-map", "0:a?",  # Include audio if available
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            output_video_path
        ])

        app.logger.info("Starting FFmpeg processing...")
        app.logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            app.logger.error(f"FFmpeg failed: {result.stderr}")
            raise Exception(f"Video processing failed: {result.stderr}")

        app.logger.info("Video generation completed successfully")

    except Exception as e:
        app.logger.error(f"Error in video generation: {e}")
        return jsonify({"error": "Video generation failed", "details": str(e)}), 500

    finally:
        # Clean up temp files (keep output video for download)
        for f_path in temp_files:
            if f_path != output_video_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception as clean_err:
                    app.logger.warning(f"Failed to clean up {f_path}: {clean_err}")

    return jsonify({
        "video_filename": os.path.basename(output_video_path)
    })

@app.route('/download-video/<filename>', methods=['GET'])
def download_video(filename):
    """Download generated video file"""
    path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "Video not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)