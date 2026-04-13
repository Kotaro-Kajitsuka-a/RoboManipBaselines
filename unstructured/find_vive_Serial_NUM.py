#!/usr/bin/env python3

import time

import numpy as np
import openvr


class ViveTracker:
    def __init__(self):
        self.vr_system = openvr.init(openvr.VRApplication_Other)

    def __del__(self):
        openvr.shutdown()

    def run(self):
        while True:
            device_data_list = self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )

            print("---- device scan ----")

            for device_idx in range(openvr.k_unMaxTrackedDeviceCount):
                self.process_device(device_idx, device_data_list)

            time.sleep(1)

    def process_device(self, device_idx, device_data_list):
        device_data = device_data_list[device_idx]

        if not device_data.bDeviceIsConnected:
            return

        device_type = self.vr_system.getTrackedDeviceClass(device_idx)

        try:
            device_sn = self.vr_system.getStringTrackedDeviceProperty(
                device_idx, openvr.Prop_SerialNumber_String
            )
        except Exception as e:
            print(
                f"Error occurred while fetching serial number for device {device_idx}: {e}"
            )
            device_sn = "Unknown"

        if device_type == openvr.TrackedDeviceClass_HMD:
            return

        if device_sn == "Null Serial Number":
            return

        is_pose_valid = device_data.bPoseIsValid

        type_map = {
            openvr.TrackedDeviceClass_GenericTracker: "Tracker",
            openvr.TrackedDeviceClass_Controller: "Controller",
            openvr.TrackedDeviceClass_TrackingReference: "BaseStation",
            openvr.TrackedDeviceClass_HMD: "HMD",
        }

        device_type_str = type_map.get(device_type, "Unknown")

        print(f"[{device_idx}] {device_type_str} | SN: {device_sn}")

        if not is_pose_valid:
            return

        pose_matrix_data = device_data.mDeviceToAbsoluteTracking

        pose_mat = np.zeros((4, 4))
        pose_mat[0:3, 0:4] = pose_matrix_data.m
        pose_mat[3, 3] = 1.0

        position = pose_mat[0:3, 3]

        # 🔥 コントローラーだけ位置表示
        if device_type == openvr.TrackedDeviceClass_Controller:
            print(f"  -> Controller Position: {position}")

        # 参考：トラッカーも見たいなら
        if device_type == openvr.TrackedDeviceClass_GenericTracker:
            print(f"  -> Tracker Position: {position}")

        # HMDも見たいなら
        if device_type == openvr.TrackedDeviceClass_HMD:
            print(f"  -> HMD Position: {position}")


if __name__ == "__main__":
    app = ViveTracker()
    app.run()
