#!/usr/bin/env python3
import os
import sys
import cv2
import shutil
import zipfile


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def get_output_paths(input_path: str, output_dir: str | None = None):
	base_dir = (
		os.path.abspath(output_dir)
		if output_dir is not None and len(str(output_dir)) > 0
		else os.path.dirname(os.path.abspath(input_path))
	)
	stem = os.path.splitext(os.path.basename(input_path))[0]
	frames_dir = os.path.join(base_dir, f"{stem}_images_tmp")
	edges_dir = os.path.join(base_dir, f"{stem}_edges_tmp")
	frames_zip = os.path.join(base_dir, f"{stem}_images.zip")
	edges_zip = os.path.join(base_dir, f"{stem}_edges.zip")
	return frames_dir, edges_dir, frames_zip, edges_zip


def extract_frames(video_path: str, frames_dir: str) -> int:
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError("Error: Could not open the video. Check file path.")

	ensure_dir(frames_dir)
	frame_index = 1
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		filename = f"{frame_index:04d}.png"
		out_path = os.path.join(frames_dir, filename)
		cv2.imwrite(out_path, frame)
		frame_index += 1

	cap.release()
	return frame_index - 1


def compute_canny_for_frames(frames_dir: str, edges_dir: str) -> int:
	ensure_dir(edges_dir)
	count = 0
	for name in sorted(os.listdir(frames_dir)):
		if not name.lower().endswith(".png"):
			continue
		in_path = os.path.join(frames_dir, name)
		image = cv2.imread(in_path)
		if image is None:
			continue
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
		edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
		base, ext = os.path.splitext(name)
		out_name = f"{base}_canny.png"
		cv2.imwrite(os.path.join(edges_dir, out_name), edges)
		count += 1
	return count


def zip_dir_contents(src_dir: str, zip_path: str) -> None:
	with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
		for name in sorted(os.listdir(src_dir)):
			full_path = os.path.join(src_dir, name)
			if os.path.isfile(full_path):
				zf.write(full_path, arcname=name)


def main() -> None:
	if len(sys.argv) not in (2, 3):
		print("Usage: python process_video.py input_video.mp4 [output_dir]")
		sys.exit(1)

	input_video = sys.argv[1]
	output_dir = sys.argv[2] if len(sys.argv) == 3 else None
	if not os.path.isfile(input_video):
		print("Error: input video file does not exist")
		sys.exit(1)

	# Make sure output dir exists if provided
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	frames_dir, edges_dir, frames_zip, edges_zip = get_output_paths(input_video, output_dir)

	# Clean previous temp dirs if exist
	for d in (frames_dir, edges_dir):
		if os.path.isdir(d):
			shutil.rmtree(d, ignore_errors=True)

	# Remove previous zips if exist
	for z in (frames_zip, edges_zip):
		if os.path.isfile(z):
			os.remove(z)

	try:
		frame_count = extract_frames(input_video, frames_dir)
		if frame_count == 0:
			print("No frames extracted from the video.")
			# Still proceed to create empty zips for consistency
			ensure_dir(edges_dir)
			zip_dir_contents(frames_dir, frames_zip)
			zip_dir_contents(edges_dir, edges_zip)
			return

		_ = compute_canny_for_frames(frames_dir, edges_dir)
		zip_dir_contents(frames_dir, frames_zip)
		zip_dir_contents(edges_dir, edges_zip)
		print(f"Done. Frames: {frame_count}")
		print(f"Created: {frames_zip}")
		print(f"Created: {edges_zip}")
	finally:
		# Cleanup temporary directories
		if os.path.isdir(frames_dir):
			shutil.rmtree(frames_dir, ignore_errors=True)
		if os.path.isdir(edges_dir):
			shutil.rmtree(edges_dir, ignore_errors=True)


if __name__ == "__main__":
	main()


