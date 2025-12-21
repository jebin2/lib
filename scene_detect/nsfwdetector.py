import cv2
import os
from transformers import pipeline
from PIL import Image
import torch

class SimpleNSFWDetector:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        print("Loading NSFW detection model...")
        self.classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=0 if torch.cuda.is_available() else -1
        )
        print("Model loaded!")
    
    def is_nsfw(self, frame):
        """Check if a frame is NSFW"""
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.classifier(frame_pil)
        
        for result in results:
            if 'nsfw' in result['label'].lower() and result['score'] > self.threshold:
                return True, result['score']
        return False, 0.0

    def is_nsfw_batch(self, frames):
        """Batch check frames for NSFW"""
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        results = self.classifier(pil_frames)  # list of results per frame
        
        outputs = []
        for res in results:
            nsfw_flag = any('nsfw' in r['label'].lower() and r['score'] > self.threshold for r in res)
            score = max([r['score'] for r in res if 'nsfw' in r['label'].lower()], default=0.0)
            outputs.append((nsfw_flag, score))
        return outputs

    
    def process_video(self, video_path, output_folder="nsfw_clips", check_interval=1, max_gap_seconds=10):
        """Process video and save frames + track time segments"""
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return
        
        os.makedirs(output_folder, exist_ok=True)
        frames_folder = os.path.join(output_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Processing: {os.path.basename(video_path)}")
        print(f"Duration: {duration:.1f}s, Max gap: {max_gap_seconds}s")
        
        # Simple tracking variables
        last_true_start_time = None
        last_false_start_time = None
        first_true_start_time = None  # Track current segment start
        nsfw_segments = []         # Store final segments
        frame_count = 0
        frames_to_check = int(fps * check_interval)  # Check every N seconds
        
        batch_size = 16
        frames_batch = []
        timestamps_batch = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frames_to_check == 0:
                timestamp = frame_count / fps
                frames_batch.append(frame)
                timestamps_batch.append(timestamp)

                # Process when batch is ready
                if len(frames_batch) == batch_size:
                    results = self.is_nsfw_batch(frames_batch)
                    for (is_nsfw, score), ts, frm in zip(results, timestamps_batch, frames_batch):
                        if is_nsfw:
                            # Time tracking logic
                            if first_true_start_time is None:
                                # Start new segment
                                first_true_start_time = ts
                                print(f"  → Started new segment at {ts:.1f}s")

                            last_true_start_time = ts

                            # Save frame
                            frame_name = f"nsfw_frame_{ts:.1f}s_score{score:.3f}.jpg"
                            frame_path = os.path.join(frames_folder, frame_name)
                            cv2.imwrite(frame_path, frm)
                            print(f"NSFW detected at {ts:.1f}s (score: {score:.3f}) - Frame saved")
                        
                        else:
                            # Not NSFW
                            last_false_start_time = ts
                            if first_true_start_time is not None:
                                time_diff = abs(last_false_start_time - last_true_start_time)
                                if time_diff > max_gap_seconds:
                                    if abs(last_true_start_time - first_true_start_time) > 1:
                                        nsfw_segments.append((first_true_start_time, last_true_start_time))
                                        self.extract_segments(video_path, [(first_true_start_time, last_true_start_time)], output_folder)
                                        print(f"  → Finished segment: {first_true_start_time:.1f}s - {last_true_start_time:.1f}s")
                                    first_true_start_time = None
                                    last_true_start_time = None
                                else:
                                    print(f"  → Waiting (diff: {time_diff:.1f}s)")
                    frames_batch, timestamps_batch = [], []  # reset
                    print(f"Progress: {timestamp:.1f}s / {duration:.1f}s", end='\r')
            
            frame_count += 1
        
        cap.release()
        
        # Handle last segment if still active
        if frames_batch:  # process remaining frames
            results = self.is_nsfw_batch(frames_batch)
            for (is_nsfw, score), ts, frm in zip(results, timestamps_batch, frames_batch):
                if first_true_start_time is not None:
                    time_diff = abs(last_false_start_time - last_true_start_time)
                    if time_diff > max_gap_seconds:
                        if abs(last_true_start_time - first_true_start_time) > 1:
                            nsfw_segments.append((first_true_start_time, last_true_start_time))
                            self.extract_segments(video_path, [(first_true_start_time, last_true_start_time)], output_folder)
                            print(f"  → Finished segment: {first_true_start_time:.1f}s - {last_true_start_time:.1f}s")
        
        # Extract clips from segments
        # if nsfw_segments:
        #     print(f"\nFound {len(nsfw_segments)} NSFW segments:")
        #     for i, (start, end) in enumerate(nsfw_segments):
        #         print(f"  Segment {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
        #     self.extract_segments(video_path, nsfw_segments, output_folder)
        # else:
        #     print("No NSFW segments found")
    
    def extract_segments(self, video_path, segments, output_folder):
        """Extract video clips for each segment"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        for i, (start_time, end_time) in enumerate(segments):
            duration = end_time - start_time
            output_path = os.path.join(output_folder, f"{video_name}_segment_{i+1}_{start_time:.0f}-{end_time:.0f}s.mp4")
            
            # Use ffmpeg to extract clip
            cmd = f'ffmpeg -ss {start_time} -i "{video_path}" -t {duration} -c copy "{output_path}" -y -loglevel quiet'
            
            if os.system(cmd) == 0:
                print(f"Saved clip: {os.path.basename(output_path)}")
            else:
                print(f"Failed to extract segment {i+1}")

# Usage
if __name__ == "__main__":
    detector = SimpleNSFWDetector(threshold=0.6)
    
    # Process single video
    video_path = "Blue is the Warmest Color 2013.mkv"
    detector.process_video(
        video_path, 
        output_folder="nsfw_clips", 
        check_interval=1,       # Check every 1 second
        max_gap_seconds=10      # Max 10 second gap within same segment
    )