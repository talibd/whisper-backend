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

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)
CORS(app)

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
                timestamp_granularities=["word"]  # <-- Add this line
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
        "segments": [s.text for s in raw_segments] if raw_segments else [], # Guard against raw_segments being None
        "words": [dict(text=w.word, start=w.start, end=w.end) for w in raw_words] if raw_words else [],
        "language": getattr(transcript_response, "language", None)
    })

@app.route('/generate-video', methods=['POST'])
def generate_video():
    file = request.files['file']
    transcript = request.form.get('transcript', "")
    words_json = request.form.get('words', None)
    keywords_json = request.form.get('keywords', None)
    broll_images_json = request.form.get('broll_images', None)
    words = json.loads(words_json) if words_json else None
    keywords = json.loads(keywords_json) if keywords_json else []
    broll_images = json.loads(broll_images_json) if broll_images_json else {}

    if not transcript:
        return jsonify({"error": "Transcript required"}), 400

    safe_filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    temp_input_path = os.path.join(tempfile.gettempdir(), f"{unique_id}_{safe_filename}")
    file.save(temp_input_path)

    class Segment:
        def __init__(self, text, start, end):
            self.start = start
            self.end = end
            self.text = text

    segments = []
    if words and len(words) > 0:
        # Group words into lines of up to 7 words each for subtitles
        group_size = 7
        for i in range(0, len(words), group_size):
            group = words[i:i+group_size]
            text = ' '.join([w['text'] for w in group])
            # Highlight keywords in this subtitle line
            text = highlight_keywords(text, keywords)
            start = group[0]['start']
            end = group[-1]['end']
            segments.append(Segment(text, start, end))
    else:
        # Fallback: split transcript into sentences with fixed duration
        import re
        sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
        duration_per_sentence = 3
        current_time = 0
        for sentence in sentences:
            if sentence.strip():
                text = highlight_keywords(sentence.strip(), keywords)
                start = current_time
                end = current_time + duration_per_sentence
                segments.append(Segment(text, start, end))
                current_time = end

                # Post-process segments to prevent overlaps and ensure minimum duration
    min_segment_duration = 0.01  # Minimum display time for a subtitle (e.g., 10ms)
    time_gap = 0.001             # Ensure at least 1ms gap between subtitles

    if segments:
        # Pass 1: Ensure each segment has at least the minimum duration.
        # This is important because subsequent adjustments might shorten segments.
        for seg in segments:
            if (seg.end - seg.start) < min_segment_duration:
                seg.end = seg.start + min_segment_duration
        
        # Pass 2: Resolve overlaps between consecutive segments.
        # Iterate n-1 times if there are n segments.
        if len(segments) > 1:
            for i in range(len(segments) - 1):
                current_seg = segments[i]
                next_seg = segments[i+1]

                # If current segment ends at or after the next one starts
                if current_seg.end >= next_seg.start:
                    # Adjust current_seg.end to be 'time_gap' before next_seg.start
                    current_seg.end = next_seg.start - time_gap
                    
                    # After adjustment, if current_seg is now shorter than min_segment_duration (or invalid),
                    # it means next_seg starts very early relative to current_seg.start.
                    # Re-enforce min_segment_duration for current_seg.
                    if (current_seg.end - current_seg.start) < min_segment_duration:
                        current_seg.end = current_seg.start + min_segment_duration
                        # If current_seg.end now >= next_seg.start again, it indicates a fundamental
                        # timing conflict in the source data that can't satisfy both non-overlap
                        # and min_duration for current_seg. The SRT will reflect this timing.

    temp_srt_path = temp_input_path + ".srt"
    output_video_path = temp_input_path + "_subtitled.mp4"

    try:
        srt_content = generate_srt(segments)
        with open(temp_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        # Prepare paths for FFmpeg, ensuring forward slashes for cross-platform compatibility
        # and proper escaping for the subtitles filter.
        ffmpeg_input_path = temp_input_path.replace('\\', '/')
        ffmpeg_output_path = output_video_path.replace('\\', '/')
        ffmpeg_srt_path = temp_srt_path.replace('\\', '/').replace(":", "\\:").replace("'", "\\'")

        # --- Image Preparation ---
        # broll_paths: maps keyword to its downloaded local image path
        # actual_broll_image_ffmpeg_inputs_args: list of FFmpeg -i arguments for unique images
        # broll_local_path_to_ffmpeg_idx: maps a local image path to its FFmpeg input index (1-based for images)
        
        # Download images and store their local paths mapped by keyword
        # (This part was mostly fine, ensuring unique filenames for downloads)
        downloaded_broll_paths_by_kw = {} # Renamed from broll_paths for clarity in this scope
        for kw, url in broll_images.items(): # broll_images from request: kw -> url
            if url:
                img_filename = f"{uuid.uuid4().hex}_{secure_filename(kw)}.jpg"
                local_img_path = os.path.join(tempfile.gettempdir(), img_filename)
                if download_image(url, local_img_path):
                    downloaded_broll_paths_by_kw[kw] = local_img_path

        actual_broll_image_ffmpeg_inputs_args = []
        broll_local_path_to_ffmpeg_idx = {} 
        current_ffmpeg_broll_input_idx = 1 # Starts at 1 (0 is main video)

        # Create FFmpeg input arguments for unique downloaded images
        unique_local_image_paths = sorted(list(set(downloaded_broll_paths_by_kw.values())))
        for local_path in unique_local_image_paths:
            actual_broll_image_ffmpeg_inputs_args.extend(["-i", local_path])
            broll_local_path_to_ffmpeg_idx[local_path] = current_ffmpeg_broll_input_idx
            current_ffmpeg_broll_input_idx += 1

        # --- Filter Complex Construction ---
        filter_complex_parts = []
        last_video_stream_label = "0:v" # Initial video stream from the main input
        
        # Collect overlay operations: (start_time, end_time, image_ffmpeg_idx)
        overlay_operations = []

        for i, segment in enumerate(segments):
             for kw, local_img_path in downloaded_broll_paths_by_kw.items():
                if kw.lower() in segment.text.lower():
                     if local_img_path in broll_local_path_to_ffmpeg_idx:
                        img_ffmpeg_idx = broll_local_path_to_ffmpeg_idx[local_img_path]
                        overlay_operations.append({
                            "start": segment.start,
                            "end": segment.end,
                            "img_idx": img_ffmpeg_idx,
                        })
        
        # Sort by start time (optional, but can be good practice)
        overlay_operations.sort(key=lambda op: op["start"])

        overlay_stream_counter = 0
        overlays_were_added_to_graph = False
        if overlay_operations:
            overlays_were_added_to_graph = True
            for op in overlay_operations:
                img_input_label = f"{op['img_idx']}:v" 
                current_overlay_output_label = f"ov{overlay_stream_counter}"
                
                overlay_filter = (
                    f"[{last_video_stream_label}][{img_input_label}]"
                    f"overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2:enable='between(t,{op['start']},{op['end']})'"
                    f"[{current_overlay_output_label}]"
                )
                filter_complex_parts.append(overlay_filter)
                last_video_stream_label = current_overlay_output_label 
                overlay_stream_counter += 1

                      # Determine the video stream to apply subtitles to.
        # If overlays were added, last_video_stream_label is the output of the last overlay.
        # If no overlays, last_video_stream_label is the original video input (e.g., "0:v").
        video_stream_for_subtitles = last_video_stream_label

        if not overlays_were_added_to_graph:
            # No overlays were added; last_video_stream_label is the original input.
            # Let's try to normalize it to a common pixel format before applying subtitles,
            # as this might help with rendering consistency, especially with HTML tags in subtitles.
            normalized_video_label = "v_normalized_for_subs"
            filter_complex_parts.append(
                f"[{last_video_stream_label}]format=yuv420p[{normalized_video_label}]"
            )
            video_stream_for_subtitles = normalized_video_label

        # Add subtitles filter to the (potentially normalized) video stream.

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
        ffmpeg_cmd.extend(actual_broll_image_ffmpeg_inputs_args) # Add all -i for b-roll images
        

        ffmpeg_cmd += [
           "-filter_complex", full_filter_complex_str,
            "-map", f"[{final_video_out_label}]", # Map the final video output of filter_complex
            "-map", "0:a?",                 # Map audio from the main video input
            "-c:a", "copy",   
            ffmpeg_output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    except Exception as e:
        print("Error in /generate-video:", e)
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_srt_path):
            os.remove(temp_srt_path)
             # Clean up downloaded b-roll images
        if 'downloaded_broll_paths_by_kw' in locals():
            for local_img_path in downloaded_broll_paths_by_kw.values():
                if os.path.exists(local_img_path):
                    try:
                        os.remove(local_img_path)
                    except Exception as clean_e:
                        print(f"Warning: Failed to remove temp b-roll image {local_img_path}: {clean_e}")
        return jsonify({"error": "Video generation failed", "details": str(e)}), 500

    finally: # General cleanup for success case (input and SRT already handled by specific try/except)
        if 'downloaded_broll_paths_by_kw' in locals(): # Ensure it was defined
            for local_img_path in downloaded_broll_paths_by_kw.values():
                if os.path.exists(local_img_path): # If not removed by exception handler
                    try: os.remove(local_img_path)
                    except: pass # Ignore errors here, already tried in except


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
        model="gpt-4.1-nano",
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
                print(f"Unsplash API error for '{kw}': {resp.status_code} {resp.text}")
                images[kw] = None
                failed_keywords.append(kw)
        except Exception as e:
            print(f"Exception fetching image for '{kw}': {e}")
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
        print(f"Failed to download {url}: {e}")
    return False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
