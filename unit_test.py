import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import logging
from _main import capture_frames, frame_generator
import warnings
import log_config
log_config.setup_logging()


class TestCaptureFramesFunction(unittest.TestCase):

    @patch('_main.cv2.imread')
    @patch('_main.cv2.VideoCapture')
    @patch('_main.frame_generator')
    def test_capture_frames_function(self, mock_imread, mock_frame_generator, mock_cv2):
        """
        Test the capture_frames function throws a warning which is expected as we are mocking files that don't exist.
        Warning: cv2.imread

        """
        cap_mock = MagicMock()
        mock_cv2.return_value = cap_mock
        cap_mock.get.return_value = 500

        # Mock frame_generator function
        frame_generator_mock = mock_frame_generator.return_value
        frame_generator_mock.side_effect = [None]  # Mock a frame returning None

        # Mock imread to simulate an exception
        mock_imread.side_effect = Exception("Simulated imread exception")

        # Redirect logging output for testing
        captured_output = StringIO()
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(captured_output))

        # Test capture_frames function
        with patch('_main.frame_generator', return_value=frame_generator_mock):
            capture_frames('input_video_path', 'output_folder_path', output_width=2, num_processes=4)

        # Reset logging
        logging.getLogger().setLevel(logging.NOTSET)
        logging.getLogger().removeHandler(logging.StreamHandler(captured_output))

        # Check for exceptions
        self.assertEqual(captured_output.getvalue(), '')

    # File Extension Test: .mp4
    @patch('_main.cv2.VideoCapture')
    @patch('_main.frame_generator')
    def test_capture_frames_mp4(self, mock_frame_generator, mock_cv2):
        cap_mock = MagicMock()
        mock_cv2.return_value = cap_mock
        cap_mock.get.return_value = 500

        # Mock frame_generator function
        frame_generator_mock = mock_frame_generator.return_value
        frame_generator_mock.return_value = [b'\x00' * (100 * 100 * 3)]  # Mock a frame

        # Redirect logging output for testing
        captured_output = StringIO()
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(captured_output))

        # Test capture_frames function with MP4 file
        capture_frames('input_video.mp4', 'output_folder_path', output_width=2, num_processes=4)

        # Reset logging to normal
        logging.getLogger().setLevel(logging.NOTSET)
        logging.getLogger().removeHandler(logging.StreamHandler(captured_output))

        warnings.filterwarnings("ignore", module="_main", category=UserWarning)
        # Check for any exceptions
        self.assertEqual(captured_output.getvalue(), '')

    # File Extension Test: .avi
    @patch('_main.cv2.VideoCapture')
    @patch('_main.frame_generator')
    def test_capture_frames_avi(self, mock_frame_generator, mock_cv2):
        cap_mock = MagicMock()
        mock_cv2.return_value = cap_mock
        cap_mock.get.return_value = 500

        # Mock frame_generator function
        frame_generator_mock = mock_frame_generator.return_value
        frame_generator_mock.return_value = [b'\x00' * (100 * 100 * 3)]  # Mock a frame

        # Redirect logging output for testing
        captured_output = StringIO()
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(captured_output))

        # Test capture_frames function with AVI file
        capture_frames('input_video.avi', 'output_folder_path', output_width=2, num_processes=4)

        # Reset logging to normal
        logging.getLogger().setLevel(logging.NOTSET)
        logging.getLogger().removeHandler(logging.StreamHandler(captured_output))

        # Check for any exceptions
        self.assertEqual(captured_output.getvalue(), '')

    # File Extension Test: .mov
    @patch('_main.cv2.VideoCapture')
    @patch('_main.frame_generator')
    def test_capture_frames_mov(self, mock_frame_generator, mock_cv2):
        cap_mock = MagicMock()
        mock_cv2.return_value = cap_mock
        cap_mock.get.return_value = 500

        # Mock frame_generator function
        frame_generator_mock = mock_frame_generator.return_value
        frame_generator_mock.return_value = [b'\x00' * (100 * 100 * 3)]  # Mock a frame

        # Redirect logging output for testing
        captured_output = StringIO()
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(captured_output))

        # Test capture_frames function with MOV file
        capture_frames('input_video.mov', 'output_folder_path', output_width=2, num_processes=4)

        # Reset logging to normal
        logging.getLogger().setLevel(logging.NOTSET)
        logging.getLogger().removeHandler(logging.StreamHandler(captured_output))

        # Check for any exceptions
        self.assertEqual(captured_output.getvalue(), '')

    def test_frame_generator_function(self):
        # Mock os.listdir to return a list of mock frame files
        with patch('os.listdir', return_value=['frame_0000.jpeg', 'frame_0001.jpeg']):
            # Call the frame_generator function
            frames = list(frame_generator('output_folder_path'))
            self.assertEqual(len(frames), 2)
            logging.info(f"frame: {frames}")


if __name__ == '__main__':
    unittest.main()
