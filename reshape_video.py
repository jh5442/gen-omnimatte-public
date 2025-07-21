import cv2
import os




test_03_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/"
                "results/gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_03_removal/test_03.mp4")
test_03_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/"
                "results/gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_03_removal/test_03_removal.mp4")
test_03_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/"
                     "results/gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_03_removal/test_03_removal_reshaped.mp4")

# test_04_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
#                          "mt_lab_test_videos/test_04.mp4")
# test_04_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_04_removal/casper_outputs/gradio_demo-2dcdf102-fg=-1-0001.mp4")
# test_04_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_04_removal/result_reshaped.mp4")

test_04_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
                         "mt_lab_test_videos/test_04.mp4")
test_04_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
                "gen_omnimatte_wan2.1_1.3B/test_04_removal.mp4")
test_04_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
                "gen_omnimatte_wan2.1_1.3B/test_04_removal_reshaped.mp4")

test_06_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
                         "mt_lab_test_videos/test_06.mp4")
test_06_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
                "gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_06_removal/casper_outputs/gradio_demo-560f6581-fg=-1-0001.mp4")
test_06_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
                "gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_06_removal/casper_outputs/result_reshaped.mp4")



def reshape_video(reference_video_path,
                  video_path,
                  output_video_path):

    # Open reference video to get the desired size
    reference_video = cv2.VideoCapture(reference_video_path)
    ret, frame_a = reference_video.read()
    print("Original video:", frame_a.shape)

    if not ret:
        raise ValueError("Cannot read reference video.")
    target_size = (frame_a.shape[1], frame_a.shape[0])  # (width, height)
    reference_video.release()

    # Open video B
    video_b = cv2.VideoCapture(video_path)

    # Prepare to write the resized video B
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or 'XVID', 'avc1', etc.
    fps = video_b.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    # Resize each frame
    while True:
        ret, frame_b = video_b.read()
        # if not ret:
        #     break
        if not ret:
            print("End of video or error reading frame.")
            break

        if frame_b is None:
            print("Warning: Frame is None.")
            continue
        resized_frame = cv2.resize(frame_b, target_size)
        out.write(resized_frame)

    video_b.release()
    out.release()


if __name__ == "__main__":
    # reshape_video(reference_video_path=test_03_original_path,
    #               video_path=test_03_path,
    #               output_video_path=test_03_save_path)

    reshape_video(reference_video_path=test_04_original_path,
                  video_path=test_04_path,
                  output_video_path=test_04_save_path)

    # reshape_video(reference_video_path=test_06_original_path,
    #               video_path=test_06_path,
    #               output_video_path=test_06_save_path)