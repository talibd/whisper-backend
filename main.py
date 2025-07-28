
# main.py - Complete Enhanced Backend with Words Per Subtitle Control
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
import math
import json
import requests
import shutil

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)
CORS(app)

# Define Segment class at module level for clarity
class Segment:
    def __init__(self, text, start, end):
        self.start = start
        self.end = end
        self.text = text

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',')

def generate_srt(segments):
    srt = ""
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip().replace('-->', '→')
        srt += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt

def get_video_dimensions(video_path: str) -> tuple[int, int] | None:
    """Gets video dimensions using ffprobe."""
    if not shutil.which("ffprobe"):
        app.logger.warning("ffprobe not found. Cannot automatically determine video dimensions.")
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
        app.logger.error(f"Error getting video dimensions with ffprobe for {video_path}: {e.stderr}")
        return None
    except Exception as e:
        app.logger.error(f"Exception in get_video_dimensions for {video_path}: {e}")
        return None

def create_subtitle_segments_from_words(words, keywords, words_per_subtitle=7):
    """Create subtitle segments by grouping words based on user preference"""
    if not words:
        return []
    
    segments = []
    words_per_subtitle = max(1, min(words_per_subtitle, 15))  # Clamp between 1-15
    
    for i in range(0, len(words), words_per_subtitle):
        group = words[i:i+words_per_subtitle]
        if not group:
            continue
            
        # Combine text from all words in the group
        text = ' '.join([w.get('text', '') for w in group])
        text = highlight_keywords(text, keywords)
        
        # Use timing from first and last word in group
        start = float(group[0].get('start', 0))
        end = float(group[-1].get('end', 0))
        
        # Ensure minimum duration
        if end - start < 0.5:
            end = start + 0.5
            
        segments.append(Segment(text, start, end))
    
    return segments

def highlight_keywords(text, keywords):
    """Highlight each keyword in the text using SRT font tags"""
    def replacer(match):
        return f"<font color=\"yellow\">{match.group(0)}</font>"
    for kw in keywords:
        pattern = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
        text = pattern.sub(replacer, text)
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
                timestamp_granularities=["segment", "word"]  # Get both segment and word timestamps
            )
        raw_segments = transcript_response.segments
        full_text = transcript_response.text

        # Get words with timestamps
        raw_words = getattr(transcript_response, "words", None)

    except Exception as e:
        print("Error in /transcribe:", e)
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return jsonify({"error": "Transcription failed", "details": str(e)}), 500

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

@app.route('/generate-video', methods=['POST'])
def generate_video():
    file = request.files['file']
    transcript_text_from_form = request.form.get('transcript', "")
    words_json = request.form.get('words', None)
    keywords_json = request.form.get('keywords', None)
    broll_images_json = request.form.get('broll_images', None)
    transcribed_segments_json = request.form.get('transcribed_segments', None)
    words_per_subtitle = int(request.form.get('words_per_subtitle', 7))  # NEW: Get words per subtitle

    keywords = json.loads(keywords_json) if keywords_json else []
    broll_images = json.loads(broll_images_json) if broll_images_json else {}
    words = json.loads(words_json) if words_json else []

    # B-roll video handling
    broll_video_files = request.files.getlist("broll_video_files")
    broll_video_config_json = request.form.get('broll_video_config', '[]')
    broll_video_config = json.loads(broll_video_config_json)
    default_broll_background_color = request.form.get('default_broll_background_color', 'black')

    # Output dimensions
    output_width_req = request.form.get('output_width', type=int)
    output_height_req = request.form.get('output_height', type=int)

    app.logger.info(f"=== GENERATE VIDEO REQUEST ===")
    app.logger.info(f"Received {len(words)} words for B-roll timing")
    app.logger.info(f"Received {len(keywords)} keywords: {keywords}")
    app.logger.info(f"Received {len(broll_images)} B-roll images: {list(broll_images.keys())}")
    app.logger.info(f"Words per subtitle: {words_per_subtitle}")

    if not transcript_text_from_form and not transcribed_segments_json:
        return jsonify({"error": "Transcript required"}), 400

    safe_filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    temp_input_path = os.path.join(tempfile.gettempdir(), f"{unique_id}_{safe_filename}")
    file.save(temp_input_path)
    temp_files_to_clean_up = [temp_input_path]

    # Process segments for subtitles - ENHANCED with words per subtitle control
    final_processing_segments = []

    if transcribed_segments_json:
        try:
            parsed_segments_from_frontend = json.loads(transcribed_segments_json)
            if isinstance(parsed_segments_from_frontend, list) and parsed_segments_from_frontend:
                for seg_data in parsed_segments_from_frontend:
                    if all(k in seg_data for k in ['text', 'start', 'end']):
                        text = highlight_keywords(seg_data['text'], keywords)
                        final_processing_segments.append(Segment(text, float(seg_data['start']), float(seg_data['end'])))
                    else:
                        app.logger.warning(f"Skipping segment due to missing keys: {seg_data}")
        except json.JSONDecodeError:
            app.logger.error("Failed to decode transcribed_segments_json.")

    # ENHANCED: Use words per subtitle setting
    if not final_processing_segments and words and len(words) > 0:
        app.logger.info(f"Creating subtitle segments with {words_per_subtitle} words per subtitle")
        final_processing_segments = create_subtitle_segments_from_words(words, keywords, words_per_subtitle)
        app.logger.info(f"Created {len(final_processing_segments)} subtitle segments")
        
    elif not final_processing_segments and transcript_text_from_form:
        # Fallback to sentence-based segmentation
        app.logger.warning("No words available, falling back to sentence-based subtitles")
        sentences = re.split(r'(?<=[.!?])\s+', transcript_text_from_form.strip())
        duration_per_sentence = 3.0
        current_time = 0.0
        for sentence_text in sentences:
            if sentence_text.strip():
                text = highlight_keywords(sentence_text.strip(), keywords)
                start = current_time
                end = current_time + duration_per_sentence
                final_processing_segments.append(Segment(text, start, end))
                current_time = end

    # Post-process segments to prevent overlaps
    min_segment_duration = 0.5  # Increased minimum duration for better readability
    time_gap = 0.1  # Increased gap for better subtitle timing

    if final_processing_segments:
        # Ensure minimum duration
        for seg in final_processing_segments:
            if (seg.end - seg.start) < min_segment_duration:
                seg.end = seg.start + min_segment_duration
                
        # Prevent overlaps
        if len(final_processing_segments) > 1:
            for i in range(len(final_processing_segments) - 1):
                current_seg = final_processing_segments[i]
                next_seg = final_processing_segments[i+1]
                if current_seg.end >= next_seg.start:
                    current_seg.end = next_seg.start - time_gap
                    if (current_seg.end - current_seg.start) < min_segment_duration:
                        current_seg.end = current_seg.start + min_segment_duration

    temp_srt_path = temp_input_path + ".srt"
    output_video_path = temp_input_path + "_subtitled.mp4"

    try:
        # Determine output video dimensions
        target_width, target_height = None, None
        if output_width_req and output_height_req:
            target_width, target_height = output_width_req, output_height_req
        else:
            dimensions = get_video_dimensions(temp_input_path)
            if dimensions:
                target_width, target_height = dimensions
            else:
                target_width, target_height = 1280, 720
                app.logger.warning(f"Using default dimensions {target_width}x{target_height}")

        srt_content = generate_srt(final_processing_segments)
        with open(temp_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        if temp_srt_path not in temp_files_to_clean_up:
             temp_files_to_clean_up.append(temp_srt_path)

        # Prepare paths for FFmpeg
        ffmpeg_input_path = temp_input_path.replace('\\', '/')
        ffmpeg_output_path = output_video_path.replace('\\', '/')
        ffmpeg_srt_path = temp_srt_path.replace('\\', '/').replace(":", "\\:").replace("'", "\\'")

        ffmpeg_additional_inputs = []
        all_input_file_paths_ordered = [ffmpeg_input_path]

        # B-roll Image Preparation
        downloaded_broll_paths_by_kw = {}
        for kw, url in broll_images.items():
            if url:
                img_filename = f"{uuid.uuid4().hex}_{secure_filename(kw)}.jpg"
                local_img_path = os.path.join(tempfile.gettempdir(), img_filename)
                temp_files_to_clean_up.append(local_img_path)
                if download_image(url, local_img_path):
                    downloaded_broll_paths_by_kw[kw] = local_img_path
                    app.logger.info(f"Downloaded B-roll image for keyword '{kw}': {local_img_path}")
                else:
                    app.logger.warning(f"Failed to download B-roll image for keyword '{kw}' from {url}")
        
        app.logger.info(f"Successfully downloaded {len(downloaded_broll_paths_by_kw)} B-roll images")
        
        broll_local_path_to_ffmpeg_idx = {}
        
        unique_local_image_paths = sorted(list(set(downloaded_broll_paths_by_kw.values())))
        for local_path in unique_local_image_paths:
            ffmpeg_additional_inputs.extend(["-i", local_path.replace('\\', '/')])
            all_input_file_paths_ordered.append(local_path)
            broll_local_path_to_ffmpeg_idx[local_path] = len(all_input_file_paths_ordered) - 1

        # B-roll Video Preparation
        saved_broll_videos_map = {}
        for broll_video_file in broll_video_files:
            original_fn = secure_filename(broll_video_file.filename)
            broll_vid_unique_id = uuid.uuid4().hex
            temp_broll_vid_path = os.path.join(tempfile.gettempdir(), f"{broll_vid_unique_id}_{original_fn}")
            broll_video_file.save(temp_broll_vid_path)
            temp_files_to_clean_up.append(temp_broll_vid_path)
            
            ffmpeg_additional_inputs.extend(["-i", temp_broll_vid_path.replace('\\', '/')])
            all_input_file_paths_ordered.append(temp_broll_vid_path)
            saved_broll_videos_map[original_fn] = {
                "path": temp_broll_vid_path,
                "ffmpeg_idx": len(all_input_file_paths_ordered) - 1
            }

        # === FIXED B-ROLL IMAGE OVERLAYS SECTION ===
        filter_complex_parts = []
        last_video_stream_label = "0:v"
        
        overlay_operations = []

        app.logger.info(f"=== B-ROLL DEBUG INFO ===")
        app.logger.info(f"Words available: {len(words) if words else 0}")
        app.logger.info(f"Keywords: {keywords}")
        app.logger.info(f"Downloaded B-roll paths: {list(downloaded_broll_paths_by_kw.keys())}")

        if words and downloaded_broll_paths_by_kw:
            app.logger.info("Using word-level timing for B-roll images")
            
            # Show first few words for debugging
            for i, word in enumerate(words[:5]):
                app.logger.info(f"Sample word {i}: '{word.get('text', '')}' at {word.get('start', 0)}-{word.get('end', 0)}s")
            
            word_matches_found = 0
            for word in words:
                word_text = str(word.get('text', '')).strip()
                # Clean word text - remove punctuation and convert to lowercase
                word_text_clean = re.sub(r'[^\w\s]', '', word_text.lower()).strip()
                word_start = float(word.get('start', 0))
                word_end = float(word.get('end', 0))
                
                # Check if this word matches any of our keywords
                for kw in keywords:
                    kw_clean = re.sub(r'[^\w\s]', '', kw.lower()).strip()
                    
                    if word_text_clean == kw_clean:
                        app.logger.info(f"🎯 MATCH FOUND: Word '{word_text}' matches keyword '{kw}' at {word_start}-{word_end}s")
                        
                        # Check if we have a B-roll image for this keyword
                        if kw in downloaded_broll_paths_by_kw:
                            local_img_path_val = downloaded_broll_paths_by_kw[kw]
                            if local_img_path_val in broll_local_path_to_ffmpeg_idx:
                                img_ffmpeg_idx = broll_local_path_to_ffmpeg_idx[local_img_path_val]
                                overlay_operations.append({
                                    "start": word_start,
                                    "end": word_end,
                                    "img_idx": img_ffmpeg_idx,
                                    "keyword": kw,
                                    "word_text": word_text
                                })
                                word_matches_found += 1
                                app.logger.info(f"✅ Added B-roll overlay for keyword '{kw}' (word: '{word_text}') at {word_start}-{word_end}s")
                                break
                            else:
                                app.logger.warning(f"❌ Image path for keyword '{kw}' not found in FFmpeg index")
                        else:
                            app.logger.warning(f"❌ No B-roll image downloaded for keyword '{kw}'")
            
            app.logger.info(f"Total word matches found: {word_matches_found}")

        elif not words and downloaded_broll_paths_by_kw:
            # Fallback to segment-based B-roll
            app.logger.warning("No word-level timing available, falling back to segment-based B-roll")
            for segment_obj in final_processing_segments:
                segment_text = segment_obj.text.lower()
                clean_segment_text = re.sub(r'<[^>]+>', '', segment_text)
                
                for kw in keywords:
                    if kw.lower() in clean_segment_text:
                        app.logger.info(f"Segment contains keyword '{kw}': {clean_segment_text[:50]}...")
                        
                        if kw in downloaded_broll_paths_by_kw:
                            local_img_path_val = downloaded_broll_paths_by_kw[kw]
                            if local_img_path_val in broll_local_path_to_ffmpeg_idx:
                                img_ffmpeg_idx = broll_local_path_to_ffmpeg_idx[local_img_path_val]
                                overlay_operations.append({
                                    "start": segment_obj.start,
                                    "end": segment_obj.end,
                                    "img_idx": img_ffmpeg_idx,
                                    "keyword": kw
                                })
                                app.logger.info(f"✅ Added segment-based B-roll overlay for keyword '{kw}' at {segment_obj.start}-{segment_obj.end}s")

        else:
            if not words:
                app.logger.warning("❌ No words data available for B-roll timing")
            if not downloaded_broll_paths_by_kw:
                app.logger.warning("❌ No B-roll images downloaded")

        # Sort operations by start time
        overlay_operations.sort(key=lambda op: op["start"])

        # Log final overlay operations
        app.logger.info(f"=== FINAL B-ROLL OVERLAYS ===")
        app.logger.info(f"Total B-roll image overlays to add: {len(overlay_operations)}")
        for i, op in enumerate(overlay_operations):
            word_info = f" (word: '{op.get('word_text', 'N/A')}')" if 'word_text' in op else ""
            app.logger.info(f"Overlay {i+1}: {op['keyword']}{word_info} from {op['start']}s to {op['end']}s")

        if len(overlay_operations) == 0:
            app.logger.error("❌ NO B-ROLL OVERLAYS WILL BE ADDED! Check the debug info above.")

        overlay_stream_counter = 0
        overlays_were_added_to_graph = False

        if overlay_operations:
            # Scale initial video stream to target dimensions
            filter_complex_parts.append(f"[{last_video_stream_label}]scale={target_width}:{target_height},setsar=1[v_scaled_main]")
            last_video_stream_label = "v_scaled_main"
            overlays_were_added_to_graph = True
            
            for op in overlay_operations:
                img_input_label = f"{op['img_idx']}:v" 
                current_overlay_output_label = f"ov{overlay_stream_counter}"
                
                overlay_filter = (
                    f"[{last_video_stream_label}][{img_input_label}]"
                    f"overlay=x=(W-w)/2:y=(H-h)/2:enable='between(t,{op['start']},{op['end']})'"
                    f"[{current_overlay_output_label}]"
                )
                filter_complex_parts.append(overlay_filter)
                last_video_stream_label = current_overlay_output_label 
                overlay_stream_counter += 1

        # If no image overlays, ensure main video is scaled
        if not overlays_were_added_to_graph and last_video_stream_label == "0:v":
            filter_complex_parts.append(f"[0:v]scale={target_width}:{target_height},setsar=1[v_scaled_main]")
            last_video_stream_label = "v_scaled_main"

        # B-roll Video Overlays (unchanged)
        broll_video_overlay_counter = 0
        for config_item in broll_video_config:
            segment_idx = config_item.get("segment_index")

            raw_original_filename = None
            possible_keys = ["originalFilename", "originalFilename ", " originalFilename"] 
            for key_to_try in possible_keys:
                if key_to_try in config_item:
                    raw_original_filename = config_item[key_to_try]
                    break
            
            if raw_original_filename is None:
                for key, value in config_item.items():
                    if "originalfilename" in key.lower().strip():
                        raw_original_filename = value
                        break

            if raw_original_filename is None or not isinstance(raw_original_filename, str) or not raw_original_filename.strip():
                app.logger.warning(f"Skipping b-roll: original_filename is invalid. Value: '{raw_original_filename}'")
                continue
                
            original_filename = secure_filename(raw_original_filename)
            
            if segment_idx is None or not original_filename or not final_processing_segments or segment_idx >= len(final_processing_segments):
                app.logger.warning(f"Skipping invalid b-roll video config item: {config_item}")
                continue

            broll_video_data = saved_broll_videos_map.get(original_filename)
            if not broll_video_data:
                app.logger.warning(f"B-roll video file '{original_filename}' not found or not processed.")
                continue
            
            segment_to_overlay_on = final_processing_segments[segment_idx]
            seg_start = segment_to_overlay_on.start
            seg_end = segment_to_overlay_on.end
            broll_ffmpeg_idx = broll_video_data["ffmpeg_idx"]
            bg_color = config_item.get("background_color", default_broll_background_color)

            scaled_broll_label = f"scaled_broll_v{broll_video_overlay_counter}"
            current_broll_output_label = f"v_after_broll_vid{broll_video_overlay_counter}"

            filter_complex_parts.append(
                f"[{broll_ffmpeg_idx}:v]setpts=PTS-STARTPTS,"
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color={bg_color}[{scaled_broll_label}]"
            )
            
            filter_complex_parts.append(
                f"[{last_video_stream_label}][{scaled_broll_label}]"
                f"overlay=0:0:enable='between(t,{seg_start},{seg_end})'[{current_broll_output_label}]"
            )
            last_video_stream_label = current_broll_output_label
            broll_video_overlay_counter += 1

        # Subtitles
        video_stream_for_subtitles = last_video_stream_label
        final_video_out_label = "v_final"
        subtitles_filter_str = f"subtitles='{ffmpeg_srt_path}'"
        filter_complex_parts.append(
            f"[{video_stream_for_subtitles}]{subtitles_filter_str}[{final_video_out_label}]"
        )

        full_filter_complex_str = ";".join(filter_complex_parts)

        # Build FFmpeg command
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", ffmpeg_input_path
        ]
        ffmpeg_cmd.extend(ffmpeg_additional_inputs)
        
        ffmpeg_cmd += [
           "-filter_complex", full_filter_complex_str,
            "-map", f"[{final_video_out_label}]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",     
            ffmpeg_output_path
        ]
        
        app.logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if process.returncode != 0:
            app.logger.error(f"FFmpeg Error: {process.stderr}")
            raise Exception(f"FFmpeg processing failed. STDERR: {process.stderr}")

    except Exception as e:
        app.logger.error(f"Error in /generate-video: {e}", exc_info=True)
        return jsonify({"error": "Video generation failed", "details": str(e)}), 500

    finally:
        for f_path in temp_files_to_clean_up:
            if os.path.exists(f_path):
                try: 
                    os.remove(f_path)
                except Exception as clean_e: 
                    app.logger.warning(f"Failed to remove temp file {f_path}: {clean_e}")

    return jsonify({
        "video_filename": os.path.basename(output_video_path)
    })

@app.route('/download-video/<filename>', methods=['GET'])
def download_video(filename):
    path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Video not found", 404

@app.route('/find-keyword', methods=['POST'])
def find_keyword():
    data = request.json
    transcript = data.get("transcript", "")
    keyword = data.get("keyword", "")

    if not transcript or not keyword:
        return jsonify({"error": "Transcript and keyword required"}), 400

    lines = transcript.split('\n')
    matches = [line for line in lines if keyword.lower() in line.lower()]

    return jsonify({"matches": matches})

@app.route('/extract-keywords', methods=['POST'])
def extract_keywords():
    data = request.json
    transcript = data.get("transcript", "")

    if not transcript:
        return jsonify({"error": "Transcript required"}), 400

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract a concise list of the most important keywords from the following transcript and don't add any keyword that is not in the transcript. Respond with a comma-separated list."},
                {"role": "user", "content": transcript}
            ]
        )
        keywords_text = response.choices[0].message.content
        keywords = []
        if keywords_text:
            keywords = [kw.strip() for kw in keywords_text.replace('\n', ',').split(',') if kw.strip()]

        return jsonify({"keywords": keywords})
    except Exception as e:
        app.logger.error(f"Error extracting keywords: {e}")
        return jsonify({"error": "Keyword extraction failed", "details": str(e)}), 500

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")


@app.route('/broll-images', methods=['POST'])
def broll_images():
    data = request.json
    keywords = data.get("keywords", [])
    if not keywords:
        return jsonify({"error": "No keywords provided"}), 400

    images = {}
    failed_keywords = []
    access_token_invalid = False

    for kw in keywords:
        url = f"https://api.unsplash.com/photos/random?query={kw}&client_id={UNSPLASH_ACCESS_KEY}&orientation=landscape"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 401:
                access_token_invalid = True
                images[kw] = None
                failed_keywords.append(kw)
            elif resp.status_code == 200:
                img_data = resp.json()
                if 'urls' in img_data and 'regular' in img_data['urls']:
                    images[kw] = img_data['urls']['regular']
                else:
                    images[kw] = None
                    failed_keywords.append(kw)
            else:
                app.logger.error(f"Unsplash API error for '{kw}': {resp.status_code} {resp.text}")
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
    return jsonify(result)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)