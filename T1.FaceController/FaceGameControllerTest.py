from FaceGameController import FaceGameController

controller = FaceGameController(debug=True,use_gpu=True)

while True:
    controller.control(use_gpu=True)