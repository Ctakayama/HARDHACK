import time
import edgeiq
import cv2

"""
Use object detection to detect objects in the frame in realtime. The
types of objects detected can be changed by selecting different models.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""

note = cv2.imread('note.jpg')
w_resize = note.shape[1]*0.4
h_resize = note.shape[0]*0.2

note = cv2.resize(note, (int(w_resize), int(h_resize)))

# This function changes labels to text string
def labelToString(dic, predictions):
    for p in predictions:
        if p.label in dic:
            p.label = dic.get(p.label)
    return predictions

# This function adds notes to image
def addNotes(image, predictions, note):
    boxes = [p.box for p in predictions]
    for b in boxes:
        image = overlayNote(image, note, b)
    return image

# This function addes single note to image
def overlayNote(image, note, box):
    x = box.end_x
    y = box.start_y
    #if image.shape[1] - y > note.shape[1] and image.shape[0] - x > note.shape[0]:
    image[y:y + note.shape[0], x - note.shape[1]:x] = note
    return image


def main():

    label_defs = {"person" : "test"}

    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/mobilenet_ssd")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                frame = edgeiq.markup_image(
                        frame, labelToString(label_defs, results.predictions), show_labels = True,
                        show_confidences = False, colors=obj_detect.colors, 
                        line_thickness = 0)
                frame = addNotes(frame, results.predictions, note)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                for prediction in results.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
