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
note = None
noteB = cv2.imread('BOTTLE.jpg')
noteC = cv2.imread('CHAIR.jpg')
noteP = cv2.imread('PERSON.jpg')
noteT = cv2.imread('TVMONITOR.jpg')

w_resizeo = noteB.shape[1]*0.2
h_resizeo = noteB.shape[0]*0.2
w_resize = noteB.shape[1]*0.2
h_resize = noteB.shape[0]*0.2

noteB = cv2.resize(noteB, (int(w_resize), int(h_resize)))
noteC = cv2.resize(noteC, (int(w_resize), int(h_resize)))
noteP = cv2.resize(noteP, (int(w_resize), int(h_resize)))
noteT = cv2.resize(noteT, (int(w_resize), int(h_resize)))

# This function changes labels to text string
def labelToString(dic, predictions):
    for p in predictions:
        if p.label in dic.keys():
            p.label = dic.get(p.label)
    return predictions

# This function adds notes to image
def addNotes(image, predictions):
    for p in predictions:
        image = overlayNote(image, p)
    return image

# This function addes single note to image
def overlayNote(image, p):
    box = p.box
    x = box.end_x
    y = box.start_y
    #if image.shape[1] - y > note.shape[1] and image.shape[0] - x > note.shape[0]:
    xi = 10
    yi = 10
    # temp = note[:,:,:]
    # for s in strToList(p.label):
    #     # cv2.putText(temp, s, (xi,yi), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,0,0), 1, cv2.LINE_AA)

    #     yi = yi + 20
    a = 25
    if(p.label == 'person'):
        note = noteP
    if(p.label == 'chair'):
        note = noteC
    if(p.label == 'tvmonitor'):
        note = noteT
    if(p.label == 'bottle'):
        note = noteB
    count = 0
    try:
        image[y:y + note.shape[0], x - note.shape[1]:x] = note

    except:
        count = count + 1
        print('failed')
        print(count)
    return image
def strToList (s):
    return s.split(' ')

def main():

    label_defs = {}

    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/mobilenet_ssd")
    obj_detect.load(engine=edgeiq.Engine.DNN)


    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=1) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:

                frame = video_stream.read()
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                frame = edgeiq.markup_image(
                        frame, labelToString(label_defs, results.predictions), show_labels = False,
                        show_confidences = False, colors=obj_detect.colors, 
                        line_thickness = 0)
                frame = addNotes(frame, results.predictions)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")
                text.append("fps:{:2.2f}".format(fps.compute_fps()))
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
