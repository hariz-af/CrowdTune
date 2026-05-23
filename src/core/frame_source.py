import cv2


class FrameSource:
    def read(self):
        raise NotImplementedError

    def release(self):
        pass

    def is_seekable(self):
        return False

    def get_duration_seconds(self):
        return 0

    def get_position_seconds(self):
        return 0

    def seek_seconds(self, _seconds):
        return False


class CameraSource(FrameSource):
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.mirror = True

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class VideoFileSource(FrameSource):
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.mirror = False

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def is_seekable(self):
        return True

    def get_duration_seconds(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0:
            return 0
        return int(frame_count / fps)

    def get_position_seconds(self):
        msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if msec > 0:
            return int(msec / 1000.0)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if fps <= 0:
            return 0
        return int(frame_pos / fps)

    def seek_seconds(self, seconds):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return False

        target_frame = max(0, int(seconds * fps))
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
