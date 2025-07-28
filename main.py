# main.py
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
        start = format_timestamp(segment.start)  # changed from segment['start']
        end = format_timestamp(segment.end)      # changed from segment['end']
        text = segment.text.strip().replace('-->', '→')  # changed from segment['text']
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
                timestamp_granularities=["segment"]  # <-- Add this line
            )
        raw_segments = transcript_response.segments # Store raw segments from API response
        full_text = transcript_response.text

        # Safely get words
         # The getattr default to None, and the conditional list comprehension later, handles 'words' correctly.
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
        "segments": [ # MODIFIED HERE to include start and end times
            {"text": s.text, "start": s.start, "end": s.end}
            for s in raw_segments
        ] if raw_segments else [],
        "words": [dict(text=w.word, start=w.start, end=w.end) for w in raw_words] if raw_words else [],
        "language": getattr(transcript_response, "language", None)
    })

@app.route('/generate-video', methods=['POST'])
def generate_video():
    file = request.files['file']
    transcript_text_from_form = request.form.get('transcript', "") # Renamed to avoid conflict
    words_json = request.form.get('words', None)  # Needed for keyword-level b-roll
    keywords_json = request.form.get('keywords', None)
    broll_images_json = request.form.get('broll_images', None)
    transcribed_segments_json = request.form.get('transcribed_segments', None)

    keywords = json.loads(keywords_json) if keywords_json else []
    broll_images = json.loads(broll_images_json) if broll_images_json else {}
    words = json.loads(words_json) if words_json else []

     # New: B-roll video handling
    broll_video_files = request.files.getlist("broll_video_files")
    broll_video_config_json = request.form.get('broll_video_config', '[]')
    broll_video_config = json.loads(broll_video_config_json)
    default_broll_background_color = request.form.get('default_broll_background_color', 'black')

    # New: Output dimensions
    output_width_req = request.form.get('output_width', type=int)
    output_height_req = request.form.get('output_height', type=int)


    if not transcript_text_from_form and not transcribed_segments_json:
        return jsonify({"error": "Transcript required"}), 400

    safe_filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    temp_input_path = os.path.join(tempfile.gettempdir(), f"{unique_id}_{safe_filename}")
    file.save(temp_input_path)
    temp_files_to_clean_up = [temp_input_path]

    # Use a clear variable name for the segments that will be used for SRT and b-roll timing
    final_processing_segments = []

    if transcribed_segments_json:
        try:
            parsed_segments_from_frontend = json.loads(transcribed_segments_json)
            if isinstance(parsed_segments_from_frontend, list) and parsed_segments_from_frontend:
                for seg_data in parsed_segments_from_frontend:
                    # Ensure essential keys are present
                    if all(k in seg_data for k in ['text', 'start', 'end']):
                        text = highlight_keywords(seg_data['text'], keywords)
                        final_processing_segments.append(Segment(text, float(seg_data['start']), float(seg_data['end'])))
                    else:
                        app.logger.warning(f"Skipping segment due to missing keys: {seg_data}")
        except json.JSONDecodeError:
            app.logger.error("Failed to decode transcribed_segments_json.")

    # Fallback logic if final_processing_segments is still empty
    if not final_processing_segments:
        app.logger.warning("Transcribed segments not available or empty, falling back to words/text.")
        if words and len(words) > 0:
            group_size = 7
            for i in range(0, len(words), group_size):
                group = words[i:i+group_size]
                text = ' '.join([w['text'] for w in group])
                text = highlight_keywords(text, keywords)
                start = float(group[0]['start'])
                end = float(group[-1]['end'])
                final_processing_segments.append(Segment(text, start, end))
        elif transcript_text_from_form: # Fallback to splitting the full transcript text
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

    # Post-process final_processing_segments to prevent overlaps and ensure minimum duration
    min_segment_duration = 0.01  # Minimum display time for a subtitle (e.g., 10ms)
    time_gap = 0.001             # Ensure at least 1ms gap between subtitles

    if final_processing_segments:
        for seg in final_processing_segments:
            if (seg.end - seg.start) < min_segment_duration:
                seg.end = seg.start + min_segment_duration
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

    # This check was removed as it was premature. temp_srt_path is added to cleanup later.
    # if os.path.exists(temp_srt_path): 
    #     temp_files_to_clean_up.append(temp_srt_path)

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
                target_width, target_height = 1280, 720 # Default fallback
                app.logger.warning(f"Using default dimensions {target_width}x{target_height}")

        srt_content = generate_srt(final_processing_segments)
        with open(temp_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        if temp_srt_path not in temp_files_to_clean_up: # Correctly add after creation
             temp_files_to_clean_up.append(temp_srt_path)
        # Prepare paths for FFmpeg, ensuring forward slashes for cross-platform compatibility
        # and proper escaping for the subtitles filter if needed (usually not for file paths).
        ffmpeg_input_path = temp_input_path.replace('\\', '/')
        ffmpeg_output_path = output_video_path.replace('\\', '/')
        ffmpeg_srt_path = temp_srt_path.replace('\\', '/').replace(":", "\\:").replace("'", "\\'")

        # --- Image Preparation ---
        # broll_paths: maps keyword to its downloaded local image path
        # actual_broll_image_ffmpeg_inputs_args: list of FFmpeg -i arguments for unique images
        # broll_local_path_to_ffmpeg_idx: maps a local image path to its FFmpeg input index (1-based for images)
        
        ffmpeg_additional_inputs = [] # For -i arguments
        all_input_file_paths_ordered = [ffmpeg_input_path] # Main video is 0

        # --- B-roll Image Preparation ---
        downloaded_broll_paths_by_kw = {}
        url_to_local_img_path = {}
        for kw, url in broll_images.items():
            if url:
                img_filename = f"{uuid.uuid4().hex}_{secure_filename(kw)}.jpg"
                local_img_path = os.path.join(tempfile.gettempdir(), img_filename)
                temp_files_to_clean_up.append(local_img_path)
                if download_image(url, local_img_path):
                    downloaded_broll_paths_by_kw[kw] = local_img_path
                    url_to_local_img_path[url] = local_img_path
        
        # The line below was moved into the loop above to prevent NameError
        # temp_files_to_clean_up.append(local_img_path) 
        broll_local_path_to_ffmpeg_idx = {} 
        

        unique_local_image_paths = sorted(list(set(downloaded_broll_paths_by_kw.values())))
        for local_path in unique_local_image_paths:
            ffmpeg_additional_inputs.extend(["-i", local_path.replace('\\', '/')])
            all_input_file_paths_ordered.append(local_path)
            broll_local_path_to_ffmpeg_idx[local_path] = len(all_input_file_paths_ordered) - 1

            # --- B-roll Video Preparation ---
        saved_broll_videos_map = {} # original_filename -> {path: temp_path, ffmpeg_idx: idx}
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

        # --- Filter Complex Construction ---
        filter_complex_parts = []
        last_video_stream_label = "0:v" # Initial video stream from the main input
        
        # B-roll Image Overlays (existing logic, adapted)
        # This part assumes images are overlaid sequentially. If simultaneous overlays are needed,
        # the filter graph needs to be structured differently (e.g. all overlays on "0:v").
        # For simplicity, let's keep sequential overlay for images first.
        overlay_operations = []

        if words:
            for word in words:
                img_url = word.get("image")
                local_img_path_val = url_to_local_img_path.get(img_url)
                if img_url and local_img_path_val and local_img_path_val in broll_local_path_to_ffmpeg_idx:
                    img_ffmpeg_idx = broll_local_path_to_ffmpeg_idx[local_img_path_val]
                    overlay_operations.append({
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "img_idx": img_ffmpeg_idx,
                    })
            if not overlay_operations:
                for kw in keywords:
                    kw_tokens = re.findall(r"\b\w+\b", kw.lower())
                    if not kw_tokens:
                        continue
                    token_count = len(kw_tokens)
                    i = 0
                    while i <= len(words) - token_count:
                        matched = True
                        for j in range(token_count):
                            word_text = re.sub(r"\W+", "", str(words[i + j].get("text", "")).lower())
                            if word_text != kw_tokens[j]:
                                matched = False
                                break
                        if matched:
                            local_img_path_val = downloaded_broll_paths_by_kw.get(kw)
                            if local_img_path_val and local_img_path_val in broll_local_path_to_ffmpeg_idx:
                                img_ffmpeg_idx = broll_local_path_to_ffmpeg_idx[local_img_path_val]
                                overlay_operations.append({
                                    "start": float(words[i]["start"]),
                                    "end": float(words[i + token_count - 1]["end"]),
                                    "img_idx": img_ffmpeg_idx,
                                })
                            i += token_count
                        else:
                            i += 1
        else:
            for segment_obj in final_processing_segments:
                for kw, local_img_path_val in downloaded_broll_paths_by_kw.items():
                    if kw.lower() in segment_obj.text.lower() and local_img_path_val in broll_local_path_to_ffmpeg_idx:
                        img_ffmpeg_idx = broll_local_path_to_ffmpeg_idx[local_img_path_val]
                        overlay_operations.append({
                            "start": segment_obj.start,
                            "end": segment_obj.end,
                            "img_idx": img_ffmpeg_idx,
                        })
        
        
        overlay_operations.sort(key=lambda op: op["start"])

        overlay_stream_counter = 0
        overlays_were_added_to_graph = False
        if overlay_operations:
            # Scale initial video stream to target dimensions if not already
            filter_complex_parts.append(f"[{last_video_stream_label}]scale={target_width}:{target_height},setsar=1[v_scaled_main]")
            last_video_stream_label = "v_scaled_main"
            overlays_were_added_to_graph = True
            for op in overlay_operations:
                img_input_label = f"{op['img_idx']}:v" 
                current_overlay_output_label = f"ov{overlay_stream_counter}"
                
                overlay_filter = (
                    f"[{last_video_stream_label}][{img_input_label}]"
                    f"overlay=x=(W-w)/2:y=(H-h)/2:enable='between(t,{op['start']},{op['end']})'" # W,H are main_w, main_h
                    f"[{current_overlay_output_label}]"
                )
                filter_complex_parts.append(overlay_filter)
                last_video_stream_label = current_overlay_output_label 
                overlay_stream_counter += 1

                   # If no image overlays, ensure main video is scaled
        if not overlays_were_added_to_graph and last_video_stream_label == "0:v":
            filter_complex_parts.append(f"[0:v]scale={target_width}:{target_height},setsar=1[v_scaled_main]")
            last_video_stream_label = "v_scaled_main"

        # B-roll Video Overlays
        broll_video_overlay_counter = 0
        for config_item in broll_video_config:
            segment_idx = config_item.get("segment_index")

            app.logger.info(f"Processing config_item: {config_item}")
            app.logger.info(f"Keys in config_item: {[repr(key) for key in config_item.keys()]}")

            # Attempt to get originalFilename, being mindful of potential key issues
            raw_original_filename = None
            # Common variations of the key name to check, in case of subtle typos or whitespace
            possible_keys = ["originalFilename", "originalFilename ", " originalFilename"] 
            for key_to_try in possible_keys:
                if key_to_try in config_item:
                    raw_original_filename = config_item[key_to_try]
                    app.logger.info(f"Found originalFilename using key: {repr(key_to_try)}")
                    break
            
            if raw_original_filename is None: # If still None after checking common variations
                # Fallback: iterate keys if direct access failed, to catch unusual key names
                for key, value in config_item.items():
                    if "originalfilename" in key.lower().strip(): # Case-insensitive and whitespace-insensitive check
                        raw_original_filename = value
                        app.logger.info(f"Found originalFilename by iterating keys (matched '{repr(key)}'): {repr(raw_original_filename)}")
                        break
            app.logger.info(f"Type of raw_original_filename: {type(raw_original_filename)}")

            # Explicitly check if it's None or not a string
            if raw_original_filename is None or not isinstance(raw_original_filename, str) or not raw_original_filename.strip():
                app.logger.warning(f"Skipping b-roll: original_filename is None, not a string, or empty. Value: '{raw_original_filename}'")
                continue
            # Now we are more confident it's a non-empty string
            original_filename = secure_filename(raw_original_filename)
            app.logger.info(f"Secured original_filename: '{original_filename}'. Assigning to Segment {segment_idx}")
            app.logger.info(f"Available b-rolls: {list(saved_broll_videos_map.keys())}")

            
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

            # Scale b-roll video to target dimensions, pad with chosen color
            filter_complex_parts.append(
                f"[{broll_ffmpeg_idx}:v]setpts=PTS-STARTPTS,"
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color={bg_color}[{scaled_broll_label}]"
            )
             # Overlay scaled b-roll video onto the last video stream
            filter_complex_parts.append(
                f"[{last_video_stream_label}][{scaled_broll_label}]"
                f"overlay=0:0:enable='between(t,{seg_start},{seg_end})'[{current_broll_output_label}]"
            )
            last_video_stream_label = current_broll_output_label
            broll_video_overlay_counter += 1

        # Subtitles
        video_stream_for_subtitles = last_video_stream_label # This is the final video stream before subtitles

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
        ffmpeg_cmd.extend(ffmpeg_additional_inputs) # Add all -i for b-roll images and videos
        

        ffmpeg_cmd += [
           "-filter_complex", full_filter_complex_str,
            "-map", f"[{final_video_out_label}]", # Map the final video output of filter_complex
            "-map", "0:a?",                 # Map audio from the main video input
            "-c:v", "libx264",              # Re-encode video
            "-preset", "medium",            # Encoding speed/quality
            "-crf", "23",                   # Constant Rate Factor (quality)
            "-c:a", "aac",                  # Re-encode audio to AAC (common)
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
                try: os.remove(f_path)
                except Exception as clean_e: app.logger.warning(f"Failed to remove temp file {f_path}: {clean_e}")


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

    # Use OpenAI to find lines with the keyword (or just do a simple search)
    # For demo, let's use a simple search:
    lines = transcript.split('\n')
    matches = [line for line in lines if keyword.lower() in line.lower()]

    # Optionally, use OpenAI for more advanced search:
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "Find all lines containing the keyword."},
    #         {"role": "user", "content": f"Transcript:\n{transcript}\n\nKeyword: {keyword}"}
    #     ]
    # )
    # matches = response['choices'][0]['message']['content']

    return jsonify({"matches": matches})

@app.route('/extract-keywords', methods=['POST'])
def extract_keywords():
    data = request.json
    transcript = data.get("transcript", "")

    if not transcript:
        return jsonify({"error": "Transcript required"}), 400

    # Use OpenAI to extract keywords
    response = openai.chat.completions.create(
        model="gpt-4.1-nano", # Consider using a more standard/available model like "gpt-3.5-turbo" or "gpt-4"
        messages=[
            {"role": "system", "content": "Extract a concise list of the most important keywords from the following transcript and don't add any keyword that is not in the transcript. Respond with a comma-separated list."},
            {"role": "user", "content": transcript}
        ]
    )
    keywords_text = response.choices[0].message.content
    # Split and clean keywords
    keywords = [] # Initialize keywords as an empty list
    if keywords_text: # Proceed only if keywords_text is not None (and implicitly not an empty string for meaningful split)
        # Split and clean keywords
        keywords = [kw.strip() for kw in keywords_text.replace('\n', ',').split(',') if kw.strip()]

    return jsonify({"keywords": keywords})

def highlight_keywords(text, keywords):
    # Highlight each keyword in the text using SRT font tags
    import re
    def replacer(match):
        return f"<font color=\"yellow\">{match.group(0)}</font>"
    for kw in keywords:
        # Use word boundaries for exact matches, case-insensitive
        pattern = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
        text = pattern.sub(replacer, text)
    return text

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")  # Set this in your .env

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
