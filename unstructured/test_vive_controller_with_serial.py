#!/usr/bin/env python3

import time

import numpy as np
import openvr
from tf_transformations import quaternion_from_matrix

# 対応付け
DEVICE_MAP = {
    "LHR-55377A2B": "right_wrist",
    "LHR-301CBF17": "left_wrist",
}


def main():
    vr_system = openvr.init(openvr.VRApplication_Other)

    try:
        while True:
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )

            for i in range(openvr.k_unMaxTrackedDeviceCount):
                pose = poses[i]

                if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
                    continue

                device_class = vr_system.getTrackedDeviceClass(i)

                if device_class != openvr.TrackedDeviceClass_Controller:
                    continue

                sn = vr_system.getStringTrackedDeviceProperty(
                    i, openvr.Prop_SerialNumber_String
                )

                if sn not in DEVICE_MAP:
                    continue

                name = DEVICE_MAP[sn]

                # 4x4行列に変換
                mat = np.eye(4)
                mat[:3, :4] = pose.mDeviceToAbsoluteTracking.m

                # 位置
                position = mat[:3, 3]

                # 姿勢（クォータニオン）
                quat = quaternion_from_matrix(mat)

                print(f"[{name}]")
                print(f"  Position: {position}")
                print(f"  Quaternion: {quat}")
                print("")

            time.sleep(1 / 30)

    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
