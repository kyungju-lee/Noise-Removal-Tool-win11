import subprocess
import sys
import os
import re
import json
import argparse
from pathlib import Path

def get_video_duration(file_path):
    """Get video duration in seconds using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'json', file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def format_duration(seconds):
    """Format seconds to readable time string"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"

def detect_silence(file_path, threshold_db, min_silence_duration=0.5):
    """Detect silent segments in video
    
    min_silence_duration: minimum duration (seconds) to be considered silence
    Default 0.5s to avoid detecting micro-pauses in speech
    """
    print(f"[INFO] 무음 구간 감지 중... (기준: {threshold_db}dB, 최소 {min_silence_duration}초)")
    sys.stdout.flush()
    
    cmd = [
        'ffmpeg', '-i', file_path, '-af',
        f'silencedetect=noise={threshold_db}dB:d={min_silence_duration}',
        '-f', 'null', '-'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    output = result.stderr if result.stderr else ''
    
    # Parse silence_start and silence_end in order they appear
    silences = []
    current_start = None
    
    for line in output.split('\n'):
        if 'silence_start' in line:
            match = re.search(r'silence_start: ([\d.]+)', line)
            if match:
                current_start = float(match.group(1))
        elif 'silence_end' in line and current_start is not None:
            match = re.search(r'silence_end: ([\d.]+)', line)
            if match:
                end = float(match.group(1))
                silences.append((current_start, end))
                current_start = None
    
    return silences

def calculate_auto_threshold(file_path):
    """Calculate automatic threshold based on audio analysis"""
    print("[INFO] 오디오 분석 중 (자동 모드)...")
    sys.stdout.flush()
    
    # Get audio stats using ffmpeg
    cmd = [
        'ffmpeg', '-i', file_path, '-af', 'astats=metadata=1:reset=1',
        '-f', 'null', '-t', '30', '-'  # Analyze first 30 seconds
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    # Default threshold for voice detection
    # -35dB is a good starting point for most speech
    threshold = -35
    print(f"[INFO] 자동 감지 기준: {threshold}dB")
    sys.stdout.flush()
    return threshold

def get_speech_segments(silences, total_duration, padding_before, padding_after):
    """Convert silence segments to speech segments with padding
    
    silences: list of (silence_start, silence_end) tuples
    We want to keep the NON-silent parts (speech segments)
    
    Timeline: [0 ... silence1_start ... silence1_end ... silence2_start ... silence2_end ... duration]
    Speech:   [0 to silence1_start] [silence1_end to silence2_start] [silence2_end to duration]
    """
    if not silences:
        return [(0, total_duration)]
    
    speech_segments = []
    current_pos = 0
    
    for silence_start, silence_end in silences:
        # Speech segment is from current_pos to silence_start (with padding)
        speech_end = silence_start + padding_after  # keep a bit of silence after speech
        
        if current_pos < speech_end:
            speech_segments.append((current_pos, min(speech_end, total_duration)))
        
        # Next speech starts at silence_end (with padding before)
        current_pos = max(0, silence_end - padding_before)
    
    # Add final segment (from last silence end to video end)
    if current_pos < total_duration:
        speech_segments.append((current_pos, total_duration))
    
    # Merge overlapping segments
    merged = []
    for seg in speech_segments:
        if seg[1] <= seg[0]:  # invalid segment
            continue
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)
    
    return merged

def create_filter_complex(segments):
    """Create ffmpeg filter_complex string for concatenation"""
    filter_parts = []
    concat_inputs = []
    
    for i, (start, end) in enumerate(segments):
        duration = end - start
        filter_parts.append(f"[0:v]trim=start={start}:duration={duration},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")
    
    concat_str = ''.join(concat_inputs)
    filter_parts.append(f"{concat_str}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    
    return ';'.join(filter_parts)

def process_video(input_path, output_path, mode, threshold, bitrate, padding_before, padding_after):
    """Main processing function"""
    print(f"[INFO] 처리 시작: {os.path.basename(input_path)}")
    sys.stdout.flush()
    
    # Get original duration
    original_duration = get_video_duration(input_path)
    print(f"[INFO] 원본 길이: {format_duration(original_duration)}")
    sys.stdout.flush()
    
    # Determine threshold
    if mode == 'auto':
        threshold_db = calculate_auto_threshold(input_path)
    else:
        threshold_db = threshold
    
    # Detect silence
    silences = detect_silence(input_path, threshold_db)
    print(f"[INFO] {len(silences)}개의 무음 구간 감지됨")
    sys.stdout.flush()
    
    if not silences:
        print("[INFO] 무음 구간이 없습니다. 원본을 복사합니다.")
        sys.stdout.flush()
        # Just re-encode with target bitrate
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-b:v', bitrate,
            '-c:a', 'aac', '-b:a', '192k',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        print(f"[SUCCESS] 출력 완료: {output_path}")
        print(f"[RESULT] 원본: {format_duration(original_duration)} | 제거: 0초 | 출력: {format_duration(original_duration)}")
        sys.stdout.flush()
        return
    
    # Get speech segments
    speech_segments = get_speech_segments(silences, original_duration, padding_before, padding_after)
    
    # Calculate output duration
    output_duration = sum(end - start for start, end in speech_segments)
    removed_duration = original_duration - output_duration
    
    print(f"[INFO] 음성 구간 {len(speech_segments)}개 추출")
    print(f"[INFO] 예상 출력 길이: {format_duration(output_duration)}")
    sys.stdout.flush()
    
    # Create filter complex
    filter_complex = create_filter_complex(speech_segments)
    
    # Build ffmpeg command
    print("[INFO] 비디오 인코딩 중...")
    sys.stdout.flush()
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter_complex', filter_complex,
        '-map', '[outv]', '-map', '[outa]',
        '-c:v', 'libx264', '-b:v', bitrate,
        '-c:a', 'aac', '-b:a', '192k',
        '-preset', 'medium',
        output_path
    ]
    
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, encoding='utf-8', errors='replace')
    
    # Monitor progress
    for line in process.stderr:
        if 'time=' in line:
            time_match = re.search(r'time=(\d{2}):(\d{2}):(\d{2})', line)
            if time_match:
                h, m, s = map(int, time_match.groups())
                current_time = h * 3600 + m * 60 + s
                progress = min(100, int((current_time / output_duration) * 100))
                print(f"[PROGRESS] {progress}%")
                sys.stdout.flush()
    
    process.wait()
    
    if process.returncode == 0:
        # Verify output
        actual_output_duration = get_video_duration(output_path)
        
        print(f"[SUCCESS] 출력 완료: {output_path}")
        print("=" * 50)
        print(f"[RESULT] 원본 길이: {format_duration(original_duration)}")
        print(f"[RESULT] 제거된 시간: {format_duration(removed_duration)}")
        print(f"[RESULT] 출력 길이: {format_duration(actual_output_duration)}")
        print("=" * 50)
        sys.stdout.flush()
    else:
        print("[ERROR] 인코딩 실패")
        sys.stdout.flush()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Silence Remover')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', help='Output video file path')
    parser.add_argument('--mode', choices=['auto', 'manual'], default='auto')
    parser.add_argument('--threshold', type=float, default=-35)
    parser.add_argument('--bitrate', default='5000k')
    parser.add_argument('--padding-before', type=float, default=0.15)
    parser.add_argument('--padding-after', type=float, default=0.3)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)
    
    process_video(
        args.input,
        args.output,
        args.mode,
        args.threshold,
        args.bitrate,
        args.padding_before,
        args.padding_after
    )

if __name__ == '__main__':
    main()
