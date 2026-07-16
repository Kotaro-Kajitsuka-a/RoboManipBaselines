# ArUco Single Marker Inpainting Comparison

Comparison artifacts for removing ArUco marker id 0 from
`0715_TrashBinRolling` / `RealXarm7DualFixedGripperDemo_world0_000.rmb`.

Common settings:

- Source video: `front_rgb_image.rmb.mp4`
- Frame size: `640x480`
- Marker mask: detected marker corners expanded around the marker center
- `expand_scale`: `1.55`
- Approximate mask margin on frame 0: `14 px` from each marker edge

`contact_sheet.jpg` compares frame 0:

- original
- OpenCV `INPAINT_TELEA`
- OpenCV `INPAINT_NS`
- OpenCV xphoto `INPAINT_FSR_FAST`
- OpenCV xphoto `INPAINT_FSR_BEST`
- OpenCV xphoto `INPAINT_SHIFTMAP`

`videos/` contains full-frame videos for the three useful methods:

- `telea_world0_000_all_frames.mp4`
- `fsr_fast_world0_000_all_frames.mp4`
- `fsr_best_world0_000_all_frames.mp4`

The final script uses `INPAINT_FSR_FAST` because it is close to
`INPAINT_FSR_BEST` visually and much faster on this dataset.
