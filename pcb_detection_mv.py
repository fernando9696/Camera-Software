# Detecting PCB using gigE camera
# program will use gigE camera to detect fiducial
import cv2 as cv
import numpy as np
import mvsdk
import platform
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = .8
RED = (0,0,255)

# Load custom model ( replace with location of .pt file)
model = YOLO('yolo_pcb_v3.pt')


def loop():
    # Enumerate devices
    DevList= mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found")
        return
    
    # List devices available and select device
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select Camera: "))
    DevInfo = DevList[i]
    print(DevInfo)

    # Turn on Camera
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("Failed to start camera({}): {}".format(e.error_Code,e.message))
        return
    
    # Get camera description
    cap = mvsdk.CameraGetCapability(hCamera)

    # Determine if camera is color or black and white
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # Black and white camera allows ISP to directly output MONO data instead of expanding to 24bit-grayscale of RGB
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # Switch camera mode to continous aquisition
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # Set manual exposure time to 10ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 10 * 1000)

    # Start SDK internal image taking thread
    mvsdk.CameraPlay(hCamera)

    # Calculate the size required for the RGB buffer
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

    # Allocate RGB buffer to store images output by ISP
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    
    while(cv.waitKey(1) & 0xFF) != ord('q'):
        # Get camera frame
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera,200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # At this point image is stored in pFrameBuffer. For color cameras pFrameBuffer=RGB data.
            # Black and white cameras store 8bit grayscale data

            # Convert pFrameBuffer into opencv image format
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            frame = cv.resize(frame, (640, 480), interpolation = cv.INTER_LINEAR)

            # TODO: Predictions go here
            detections = model(frame)[0]
            prediction = model.predict(frame)[0]

            annotated_frame = detections.plot()

            # Filter weaker detections
            for data in prediction.boxes.data.tolist():
                confidence = data[4]
            
                if float(confidence) < CONFIDENCE_THRESHOLD:
                    continue

                # if confidence > CONFIDENCE_THRESHOLD
                # Change to circle?
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), RED, thickness=2)
            
                # Calculate center point
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2

                coordinates = f'Center: ({center_x}, {center_y})'
                cv.putText(frame, coordinates, (10,30), cv.FONT_HERSHEY_SIMPLEX, .5, RED, 2, cv.LINE_AA)
                cv.circle(frame, (center_x, center_y), radius=2, color=RED, thickness=-1)


            cv.imshow("Filtered Frame", frame)
            cv.imshow("Unfiltered Frame", annotated_frame)
        
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    # Close camera
    mvsdk.CameraUnInit(hCamera)

    # Release frame buffer
    mvsdk.CameraAlignFree(pFrameBuffer)

def main():
    try:
        loop()
    finally:
        cv.destroyAllWindows()

main()
