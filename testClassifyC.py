from ctypes import CDLL, c_double, c_int, c_uint8
import cv2
import numpy as np
import timeit


def main():
    img_path = "/home/agustin/Code/Matlab/ClasesDoc/AlgoritmosGeneticos/Pictures/c1/img_048.png"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (50, 50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    so_file = "/home/agustin/Code/Python/CFunctionsInPython/classifyImage.so"
    processing = CDLL(so_file)
    processing.classifyImage.restype = c_double
    seq = c_uint8 * 7500

    img_v = img.flatten("F")

    y = processing.classifyImage(seq(*(img_v)))
    print(y)


def camera():
    so_file = "/home/agustin/Code/Python/CFunctionsInPython/classifyImage.so"
    processing = CDLL(so_file)
    processing.classifyImage.restype = c_double
    seq = c_uint8 * 7500

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
    while True:
        cropped_images = []
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped_images.append(frame[0:50, 0:50])
        cropped_images.append(frame[50:100, 0:50])
        cropped_images.append(frame[0:50, 50:100])
        cropped_images.append(frame[50:100, 50:100])
        cropped_images.append(frame[0:50, 100:150])
        cropped_images.append(frame[50:100, 100:150])
        cropped_images.append(frame[0:50, 150:200])
        cropped_images.append(frame[50:100, 150:200])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow("frame", gray)
        cv2.imshow("medio", cropped_images[5])
        for img in cropped_images:
            img_v = img.flatten("F")

            y = processing.classifyImage(seq(*(img_v)))
            print(f"{y:.2f}", end="    q")
        print()

        if cv2.waitKey(1) == ord("q"):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def procesar_imagen_matlab(
    file: str = "/home/agustin/Code/Matlab/ClasesDoc/AlgoritmosGeneticos/Pictures/c1/img_048.png",
) -> float:
    img = cv2.imread(file)
    img = cv2.resize(img, (50, 50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    so_file = "/home/agustin/Code/Python/CFunctionsInPython/classifyImage.so"
    processing = CDLL(so_file)
    processing.classifyImage.restype = c_double
    seq = c_uint8 * 7500

    img_v = img.flatten("F")

    y = processing.classifyImage(seq(*(img_v)))
    return y


# Load a model imported from Tensorflow
tensorflow_net = cv2.dnn.readNetFromTensorflow("/home/agustin/Code/Python/Model_LP2021/model_april_20_frozen.pb")


def procesar_imagen_tensorflow(
    file: str = "/home/agustin/Code/Matlab/ClasesDoc/AlgoritmosGeneticos/Pictures/c1/img_048.png",
) -> float:

    # Input image
    img = cv2.imread(file)

    # Use the given image as input, which needs to be blob(s).
    tensorflow_net.setInput(cv2.dnn.blobFromImage(img, size=(128, 96), swapRB=True, crop=False))

    # Runs a forward pass to compute the net output
    network_output = tensorflow_net.forward()

    return np.argmax(network_output)


if __name__ == "__main__":
    number = 1_000
    classes = ["banistas", "Negras", "Verdes", "Metas"]
    # camera()
    # main()
    print(procesar_imagen_matlab())
    print(classes[procesar_imagen_tensorflow()])

    print(timeit.timeit(procesar_imagen_matlab, number=number))
    print(timeit.timeit(procesar_imagen_tensorflow, number=number))
