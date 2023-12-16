import argparse
import cv2
import logging
import os
import time
import numpy as np
from tqdm import tqdm
import log_config

log_config.setup_logging()


def frame_generator(output_folder):
    # Generator to yield frames one by one
    frame_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".jpeg")])
    for f in frame_files:
        yield cv2.imread(os.path.join(output_folder, f))


def capture_frames(input_video, output_folder, output_width=1080):
    cap = None
    try:
        cap = cv2.VideoCapture(input_video)

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Progress bar ANSI colors
        green = "\033[92m"
        lime = "\033[38;5;154m"
        reset = "\033[0m"

        with tqdm(total=total_frames, desc=f"{green}Processing frames: {reset}", unit="frames",
                  bar_format=f"{{desc}}{green}{{percentage:.1f}}%|"
                             f"{lime}{{bar}}{reset}{green}{{r_bar}}{reset}") as pbar:

            # Iterate over frames using generator
            frames = []
            for frame_number, frame in enumerate(frame_generator(output_folder)):
                if frame_number >= total_frames:
                    break

                # Aspect ratio
                aspect_ratio = original_width / original_height

                # Resize and convert to RGB
                resized_frame = cv2.resize(frame, (output_width, int(output_width / aspect_ratio)))
                resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # Create output path for processed frame
                output_path = os.path.join(output_folder, f"frame_{frame_number:04d}.jpeg")

                try:
                    # Save resized frame
                    cv2.imwrite(output_path, resized_frame_rgb)

                    # Add converted frame to the frames list and update
                    frames.append(resized_frame_rgb)
                    pbar.set_postfix(frames_processed=frame_number + 1)
                    pbar.update()

                except Exception as e:
                    logging.error(f"Error in processing frame {frame_number}: {str(e)}")

    finally:
        cap.release()

    return frames


def concat_frames(frames, output_path, original_width, output_format="jpeg", image_quality=100):
    # Concatenate frames and save the compiled image
    try:
        avg_aspect_ratio = sum(frame.shape[1] / frame.shape[0] for frame in frames) / len(frames)
        default_output_width = 3840

        # Calculate output height and dynamic output width
        output_height = int(default_output_width / avg_aspect_ratio)
        dynamic_output_width = int(original_width * (output_height / frames[0].shape[0]))
        logging.info(f"Average Aspect Ratio: {avg_aspect_ratio}\nDynamic Output Width: {dynamic_output_width}")

        # Concatenate frames sequentially, resize, and save compiled image with the specified format
        output_image = cv2.hconcat(frames)
        output_image = cv2.resize(output_image, (dynamic_output_width, output_height))

        # Determine the file extension based on output format
        file_extension = output_format.lower()
        if file_extension not in ["jpeg", "jpg", "png", "tiff"]:
            logging.warning(f"Unsupported output format '{output_format}', using 'jpeg' instead.")
            file_extension = "jpeg"

        # Save the image with the specified format and quality
        output_path_with_extension = f"{os.path.splitext(output_path)[0]}.{file_extension}"
        cv2.imwrite(output_path_with_extension, output_image, [cv2.IMWRITE_JPEG_QUALITY, image_quality])

    except Exception as e:
        logging.error(f"Error in concatenating frames: {str(e)}")


def parse_arguments(usage_help=False):
    parser = argparse.ArgumentParser(description="Process video frames and concatenate them.")
    parser.add_argument("output_folder", default="frames_output", nargs="?",
                        help="Path to the output folder to save frames")
    parser.add_argument("input_video", nargs="?", default=None,
                        help="Path to the input video file. Include the file extension (ex: .mp4)")
    parser.add_argument("--output_image_path", type=str, default="compiled_img.jpeg",
                        help="Path for the compiled image.")
    parser.add_argument("--output_width", type=int, default=2,
                        help="Width of the output frames ( default is 2 )")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of processes to use for saving frames in parallel")
    parser.add_argument("--output_format", type=str, default="jpeg",
                        help="Output image format (e.g., jpeg, png, tiff)")
    parser.add_argument("--image_quality", type=int, default=100,
                        help="Image quality for compressed formats (e.g., JPEG)")
    parser.add_argument("--video_extensions", nargs="+", default=[".mp4", ".mov", ".avi", ".mkv"],
                        help="Supported video file extensions")

    if usage_help:
        print("Usage: python _main.py [output_folder_path] [input_video_path] "
              "[--output_width 2] [--output_image_path compiled_img.jpeg] [--num_processes 4] [--output_format jpeg] "
              "[--image_quality 95] [--video_extensions .mkv]")

    return parser.parse_args()


def find_video_in_directory(video_extensions=None):
    # Check for video files in the current directory and use the first file found
    if video_extensions is None:
        video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    video_files = [file for file in os.listdir() if file.endswith(tuple(video_extensions))]
    if video_files:
        return video_files[0]
    return None


def main():
    try:
        start_time = time.time()
        args = parse_arguments()

        # Input arguments
        input_video_path = args.input_video
        if input_video_path is None:
            input_video_path = find_video_in_directory(video_extensions=args.video_extensions)

        if input_video_path is None:
            logging.error("Error: No video file found in the current directory.")
            return

        # Convert to absolute paths
        output_folder_path = os.path.abspath(args.output_folder)
        output_image_path = os.path.abspath(args.output_image_path)

        # Processing and output arguments
        output_width = args.output_width

        # Create a folder for all output frames
        os.makedirs(output_folder_path, exist_ok=True)

        frames = capture_frames(input_video_path, output_folder_path, output_width=output_width)

        if not frames:
            logging.error("Error: No frames captured.")
            return

        original_width = frames[0].shape[1]

        # Convert frames list to a numpy array and concat frames to create a compiled image
        frames = np.stack(frames)
        concat_frames(frames, output_image_path, original_width, output_format=args.output_format,
                      image_quality=args.image_quality)

        # Calculate video duration and get codec
        cap = cv2.VideoCapture(input_video_path)
        codec_info = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = ''.join([chr((codec_info >> 8 * i) & 0xFF) for i in range(4)])
        logging.info(f"Video Codec: {fourcc_str}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()

        formatted_minutes = int(video_duration / 60)
        formatted_seconds = int(video_duration % 60)

        end_time = time.time()
        total_time = end_time - start_time

        logging.info(f"Total processing time: {total_time:.2f} seconds\n"
                     f"Video duration: {video_duration:.2f} seconds ({formatted_minutes}m {formatted_seconds}s)")

    except FileNotFoundError as file_not_found_error:
        logging.error(f"File not found: {str(file_not_found_error)}")
    except cv2.error as cv2_error:
        logging.error(f"OpenCV error: {str(cv2_error)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
