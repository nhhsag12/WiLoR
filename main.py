import cv2
import torch
# from camera.realsense_stream import RealSenseCamera
from camera_manager import CameraManager
from hand_model import HandModel

def main():
    # Initialize the RealSense camera
    # camera = RealSenseCamera()
    # camera.start_stream()
    camera_manager = CameraManager()
    
    # Load the hand reconstruction model
    model = HandModel(
        checkpoint_path='pretrained_models/wilor_final.ckpt',
        cfg_path='pretrained_models/model_config.yaml'
    )

    try:
        if not camera_manager.start_stream():
            print("Failed to start camera stream.")
            return
        print("Camera stream started successfully.")

        while True:
            # Capture a frame from the camera
            # frame = camera.capture_frame()
            frame = camera_manager.get_frames()
            if frame is None:
                break

            new_width = 96
            new_height = 96
            dsize= (new_width, new_height)

            # Resize the image using INER_AREA for downscaling
            resized_frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)

            # Perform inference to get hand pose
            output_frame, _ = model.render_reconstruction(resized_frame, conf=0.3)

            # Display the output frame
            cv2.imshow("Hand Reconstruction", output_frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop the camera stream and release resources
        camera_manager.stop_stream()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()