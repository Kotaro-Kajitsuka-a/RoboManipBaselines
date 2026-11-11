import os
import time
import unittest

from flexible_pressure_sensor_sample.pressure_sensor_server import (
    PressureSensorServer,
)


class TestPressureSensorServer(unittest.TestCase):
    def test_receives_sensor_data_for_20_seconds(self):
        port = os.environ.get("PRESSURE_SENSOR_PORT", "/dev/ttyACM0")
        server = PressureSensorServer(port, timeout=0.1)
        distinct_frame_count = 0
        previous_frame_no = None

        try:
            server.start()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                self.assertTrue(
                    server.is_alive(),
                    f"Pressure sensor server stopped: {server.stop_reason}",
                )
                latest_frame = server.get_latest_frame()
                self.assertIsNotNone(latest_frame, "No pressure sensor frame received.")

                frame_no, matrix = latest_frame
                self.assertEqual(matrix.ndim, 2)
                self.assertGreater(matrix.size, 0)
                if frame_no != previous_frame_no:
                    distinct_frame_count += 1
                    previous_frame_no = frame_no
                time.sleep(0.02)

            self.assertTrue(
                server.is_alive(),
                f"Pressure sensor server stopped: {server.stop_reason}",
            )
            self.assertGreater(
                distinct_frame_count, 1, "Pressure sensor frames did not advance."
            )
            print(f"Received {distinct_frame_count} distinct frames in 20 seconds.")
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
