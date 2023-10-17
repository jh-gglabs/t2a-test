from zeggs.zeggs_inference import ZeroEggsInference

network_dir = "./assets/zeggs/saved_models/"
config_dir = "./assets/zeggs/processed_v1/"

zeggs_module = ZeroEggsInference(
    network_directory_path=network_dir,
    config_directory_path=config_dir
)

# Style Encoding Type
## "Example" based
audio_file_path = "./samples/067_Speech_2_x_1_0.wav"
bvh_file_path = "./samples/067_Speech_2_x_1_0.bvh"

bvh_output = zeggs_module.generate_gesture_from_audio_file(
    audio_file_path, bvh_file_path, style_encoding_type="example"
)

## "Label" based
# bvh_output = zeggs_module.generate_gesture_from_audio_file(
    # audio_file_path, bvh_file_path, style_encoding_type="label", style_name="Sad"
# )

# bvh.save("./test.bvh", bvh_output)
