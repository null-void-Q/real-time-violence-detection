import time

import gradio as gr

from controller import Controller, ModelConfig, StartUpConfig

# Initialize the controller
controller = Controller()


def start(video, clip_size, threshold, memory, input_format):
    source = input_format
    if input_format == "Video Upload":
        source = video

    if source is None:
        raise gr.Error("Error: No input video.", duration=2)

    config = StartUpConfig(
        source=source,
        modelConfig=ModelConfig(
            clip_size=clip_size, memory=memory, threshold=threshold
        ),
    )

    controller.start(config)

    for frame, label in controller.frame_stream():
        yield (
            gr.update(value=frame),
            gr.update(
                value=format_full_label(label), elem_classes=[label2color(label)]
            ),
            str(round(controller.get_playback_FPS())) + " Frames/s",
            str(round(controller.frame_rate)) + " Frames/s",
            str(round(controller.streaming_delay)) + " Seconds",
            str(round(controller.get_capture_FPS())) + " Frames/s",
        )


def handle_webcam_stream(frame):
    if not controller.video_capture:
        time.sleep(0.1)
        return gr.update(value=1)

    controller.trigger_capture(frame)


def end_processing():
    controller.end()
    return (
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
    )


def on_tab_select(event_data: gr.SelectData):
    selected_tab = event_data.value
    return selected_tab


def label2color(label):
    if label["label"] == "NonViolence":
        return "green-text"
    else:
        return "red-text"


def format_full_label(label):
    labels = {}
    if label["label"] == "NonViolence":
        labels["NonViolence"] = label["score"] / 100
        # labels["Violence"] = 1 - labels["NonViolence"]
    else:
        labels["Violence"] = label["score"] / 100
        # labels["NonViolence"] = 1 - labels["Violence"]
    return labels


# ---------------

css = """
    .red-text .output-class {color: red !important;}
    .green-text .output-class {color: green !important;}
"""
# Create Gradio Interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("## Violence Detection Demo")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs() as tabs:
                with gr.Tab("Video Upload"):
                    video_input = gr.Video(
                        label="Upload Video",
                        sources=["upload"],
                        format="mp4",
                        include_audio=False,
                        height=480,
                    )
                    with gr.Row():
                        start_button = gr.Button("Start Processing", scale=2)
                        end_button = gr.Button("End Processing", scale=1)
                with gr.Tab("Webcam Streaming"):
                    webcam_input = gr.Image(
                        label="Webcam Stream",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        height=480,
                    )
                    # with gr.Row():
                    # cam_start_button = gr.Button("Start Processing",scale=2)
                    # cam_end_button = gr.Button("End Processing",scale=1)

            clip_size_dropdown = gr.Dropdown(
                choices=[16, 32, 64, 128],
                value=32,
                label="Clip Size",
                info="the number of frames the model will look at per  prediction.",
            )
            memory_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="Memory",
                info="the number of predictions to average before making the final prediction.",
            )
            threshold_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=65,
                step=1,
                label="Threshold (%)",
                visible=True,
                info="minimum % confidence required for labeling a clip as 'Violent'",
            )

            gr.Examples(
                examples=[["sample_video.mp4"]],
                inputs=video_input,
                label="Click on a sample video to load it:",
                cache_examples=False,
            )

        with gr.Column(scale=1):
            # Output Components
            with gr.Group():
                video_output = gr.Image(
                    label="Processing Feed",
                    type="numpy",
                    height=512,
                    format="jpeg",
                    show_download_button=False,
                    interactive=False,
                    show_fullscreen_button=False,
                )
                vid_label = gr.Label()
            # Status and Metrics
            with gr.Group():
                with gr.Row():
                    playback_fps_output = gr.Textbox(
                        label="Playback FPS", value="", interactive=False
                    )
                    fps_output = gr.Textbox(
                        label="Processing Speed", value="", interactive=False
                    )
                with gr.Row():
                    delay_output = gr.Textbox(
                        label="Processing Delay", value="", interactive=False
                    )
                    capture_fps = gr.Textbox(
                        label="Video Capture Speed", value="", interactive=False
                    )

            gr.Markdown("""
                ### **Note:** This project employs a model with Limited scope and abilities, intended only for demonstration purposes.
                """)

        selected_tab = gr.Textbox(
            label="Selected Tab", value="Video Upload", visible=False
        )
        webcam_state = gr.Number(value=0, visible=False)  # gr.State(value=False)
        tabs.select(on_tab_select, None, selected_tab)

        processing_event = start_button.click(
            fn=start,
            inputs=[
                video_input,
                clip_size_dropdown,
                threshold_slider,
                memory_slider,
                selected_tab,
            ],
            outputs=[
                video_output,
                vid_label,
                playback_fps_output,
                fps_output,
                delay_output,
                capture_fps,
            ],
        )

        end_button.click(
            fn=end_processing,
            inputs=None,
            outputs=[
                video_output,
                vid_label,
                playback_fps_output,
                fps_output,
                delay_output,
                capture_fps,
                video_input,
            ],
            cancels=[processing_event],
        )

        cam_stream = webcam_input.stream(
            handle_webcam_stream,
            inputs=[webcam_input],
            outputs=[webcam_state],
            time_limit=1000,
            stream_every=0.033,
        )

        output_stream = gr.on(
            triggers=[webcam_state.change],
            fn=start,
            inputs=[
                video_input,
                clip_size_dropdown,
                threshold_slider,
                memory_slider,
                selected_tab,
            ],
            outputs=[
                video_output,
                vid_label,
                playback_fps_output,
                fps_output,
                delay_output,
                capture_fps,
            ],
            trigger_mode="once",
            concurrency_limit=1,
        )
        cam_stream.then(
            fn=end_processing,
            inputs=None,
            outputs=[
                video_output,
                vid_label,
                playback_fps_output,
                fps_output,
                delay_output,
                capture_fps,
                video_input,
            ],
            cancels=[processing_event],
        )
        print("* Running on local URL:  http://127.0.0.1:7860")
        print("* Running on local URL:  http://localhost:7860")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, quiet=True)
