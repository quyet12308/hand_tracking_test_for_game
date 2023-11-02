import mediapipe as mp
import cv2

# Khởi tạo mô-đun Mediapipe cho việc nhận dạng bàn tay
mp_drawing_util = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Khởi tạo đối tượng Mediapipe Hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Khởi tạo đối tượng VideoCapture để lấy video từ webcam
cap = cv2.VideoCapture(0)

while True:
    # Đọc một khung hình từ video
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi không gian màu từ BGR sang RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý khung hình để nhận dạng bàn tay
    results = hands.process(image_rgb)

    # Kiểm tra xem có bàn tay trong khung hình hay không
    if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
        #     # Lấy tọa độ các điểm đặc trưng của bàn tay
        #     thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        #     index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        #     middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        for hand_landmarks in results.multi_hand_landmarks:
            # Lấy tọa độ các điểm đặc trưng của bàn tay
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Xác định các cử chỉ điều khiển dựa trên tọa độ của các điểm đặc trưng
            if index_finger_tip.x < thumb_tip.x and middle_finger_tip.x < thumb_tip.x:
                # Di chuyển nhân vật sang trái
                print("Di chuyển sang trái")
            elif index_finger_tip.x > thumb_tip.x and middle_finger_tip.x > thumb_tip.x:
                # Di chuyển nhân vật sang phải
                print("Di chuyển sang phải")
            elif middle_finger_tip.y > thumb_tip.y:
                # Di chuyển nhân vật về phía trước
                print("Di chuyển về phía trước")
            elif thumb_tip.x < index_finger_tip.x and thumb_tip.x < middle_finger_tip.x:
                # Di chuyển nhân vật lùi lại phía sau
                print("Lùi lại phía sau")

            # Vẽ các điểm đặc trưng của bàn tay lên khung hình
            mp_drawing_util.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị khung hình đã xử lý lên màn hình
    cv2.imshow('Hand Gesture Control', frame)

    # Thoát vòng lặp khi nhấn phím Esc
    if cv2.waitKey(1) == 27:
        break

# Giải phóng tài nguyên và kết thúc chương trình
cap.release()
cv2.destroyAllWindows()