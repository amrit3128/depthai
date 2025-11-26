import depthai as dai

pipeline = dai.Pipeline()
camRgb = pipeline.createColorCamera()
print("createColorCamera is working!")
