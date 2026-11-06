import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from robo_manip_baselines.misc.VisualizeData import VisualizeData


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_filename", type=str)
    parser.add_argument("--skip", default=1, type=int, help="skip")
    parser.add_argument(
        "-o",
        "--output_mp4_filename",
        type=str,
        help="save result as mp4 file when this option is set",
    )
    parser.add_argument("--mp4_codec", type=str, default="mp4v")
    parser.add_argument(
        "--rgb_crop_size_list",
        type=int,
        nargs="+",
        default=None,
        help="List of RGB crop sizes (width, height).",
    )
    parser.add_argument(
        "--use_pointcloud_camera",
        action="store_true",
        help="Use pointcloud cameras instead of camera_names.",
    )
    parser.add_argument(
        "--display_stored_pointcloud",
        action="store_true",
        help="Display the point cloud stored in the log file.",
    )
    parser.add_argument(
        "--sheet_key",
        type=str,
        default="sheet",
        help="Data key of the recorded sheet tactile sensor.",
    )
    return parser.parse_args()


class VisualizeSheetTactileData(VisualizeData):
    def __init__(self, sheet_key="sheet", **kwargs):
        self.sheet_key = sheet_key
        super().__init__(**kwargs)

    def setup_plot(self):
        super().setup_plot()
        if not self.camera_names:
            raise ValueError(
                "Sheet visualization needs at least one camera row for the tactile image."
            )

        if self.sheet_key not in self.data_manager.all_data_seq:
            raise KeyError(
                f"Sheet key '{self.sheet_key}' was not found in the RMB data."
            )

        sheet_data = self.data_manager.get_data_seq(self.sheet_key)
        if sheet_data.ndim != 3 or len(sheet_data) == 0:
            raise ValueError(
                f"Expected '{self.sheet_key}' to contain a non-empty sequence of 2D matrices."
            )
        self.sheet_image = None
        self.sheet_ax = self.fig.add_subplot(len(self.camera_names) + 1, 4, 4 * (1 + 1))
        self.ax[1, 3] = self.sheet_ax
        self.sheet_ax.set_title(f"{self.sheet_key} tactile", fontsize=12)

        self.fig.set_layout_engine(None)
        self.fig.subplots_adjust(bottom=0.14)
        self.is_playing = True
        self.current_frame_idx = 0
        self.pending_frame_idx = None
        self.frame_count = len(sheet_data)
        self.play_button_ax = self.fig.add_axes([0.43, 0.035, 0.12, 0.045])
        self.prev_button_ax = self.fig.add_axes([0.30, 0.035, 0.10, 0.045])
        self.next_button_ax = self.fig.add_axes([0.58, 0.035, 0.10, 0.045])
        self.slider_ax = self.fig.add_axes([0.73, 0.045, 0.22, 0.025])
        self.play_button = Button(self.play_button_ax, "Pause")
        self.prev_button = Button(self.prev_button_ax, "Previous")
        self.next_button = Button(self.next_button_ax, "Next")
        self.frame_slider = Slider(
            self.slider_ax,
            "Frame",
            0,
            self.frame_count - 1,
            valinit=0,
            valstep=1,
        )
        self.play_button.on_clicked(self.toggle_play)
        self.prev_button.on_clicked(self.step_backward)
        self.next_button.on_clicked(self.step_forward)
        self.frame_slider.on_changed(self.seek_frame)

    def toggle_play(self, _event=None):
        self.is_playing = not self.is_playing
        self.play_button.label.set_text("Pause" if self.is_playing else "Play")

    def step_backward(self, _event=None):
        self.is_playing = False
        self.play_button.label.set_text("Play")
        self.pending_frame_idx = max(self.current_frame_idx - self.skip, 0)

    def step_forward(self, _event=None):
        self.is_playing = False
        self.play_button.label.set_text("Play")
        self.pending_frame_idx = min(
            self.current_frame_idx + self.skip, self.frame_count - 1
        )

    def seek_frame(self, frame_idx):
        self.is_playing = False
        self.play_button.label.set_text("Play")
        self.pending_frame_idx = int(frame_idx)

    def key_event(self, event):
        if event.key in ("q", "escape"):
            self.quit_flag = True
        elif event.key == " ":
            self.toggle_play()
        elif event.key in ("left", "a"):
            self.step_backward()
        elif event.key in ("right", "d"):
            self.step_forward()

    def get_frame_indices(self):
        frame_count = len(self.data_manager.get_data_seq("time"))
        previous_frame_idx = None
        next_frame_idx = 0

        while next_frame_idx < frame_count:
            while not self.is_playing and self.pending_frame_idx is None:
                if not plt.fignum_exists(self.fig.number) or self.quit_flag:
                    return
                plt.pause(0.05)

            if self.pending_frame_idx is not None:
                next_frame_idx = self.pending_frame_idx
                self.pending_frame_idx = None

            if previous_frame_idx is not None:
                expected_next = previous_frame_idx + self.skip
                if next_frame_idx != expected_next:
                    for key in self.data_list:
                        if key in self.data_manager.all_data_seq:
                            self.data_list[key] = [
                                self.data_manager.get_single_data(key, idx)
                                for idx in range(0, next_frame_idx, self.skip)
                            ]

            self.current_frame_idx = next_frame_idx
            self.frame_slider.eventson = False
            self.frame_slider.set_val(next_frame_idx)
            self.frame_slider.eventson = True
            yield next_frame_idx

            previous_frame_idx = next_frame_idx
            next_frame_idx = previous_frame_idx + self.skip

    def wait_after_frame(self):
        interval = 0.1
        elapsed = 0.0
        while self.is_playing and elapsed < interval:
            if not plt.fignum_exists(self.fig.number) or self.quit_flag:
                return
            step = min(0.02, interval - elapsed)
            plt.pause(step)
            elapsed += step

    def handle_rgb_image(self, camera_idx, time_idx, rgb_key):
        rgb_image = super().handle_rgb_image(camera_idx, time_idx, rgb_key)
        if camera_idx == 0:
            if self.sheet_image is None:
                self.sheet_image = self.sheet_ax.imshow(
                    self.data_manager.get_single_data(self.sheet_key, time_idx)[
                        ::-1, :
                    ],
                    cmap="turbo",
                    vmin=0,
                    vmax=255,
                    interpolation="nearest",
                )
                self.fig.colorbar(self.sheet_image, ax=self.sheet_ax)
            else:
                self.sheet_image.set_data(
                    self.data_manager.get_single_data(self.sheet_key, time_idx)[::-1, :]
                )
        return rgb_image


def main():
    args = vars(parse_argument())
    sheet_key = args.pop("sheet_key")
    viz = VisualizeSheetTactileData(sheet_key=sheet_key, **args)
    viz.plot()


if __name__ == "__main__":
    main()
