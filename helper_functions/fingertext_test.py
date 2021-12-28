from main import *


def worker_predict_image():
    counter = 1

    while True:


        frame = cv2.imread(f"finger_text_selected/camera{counter}.jpg", flags=cv2.IMREAD_COLOR)

        if frame is None:
            counter += 1
            continue


        start_time = time.time()

        _, this_frame = cv2.imencode('.jpg', frame)
        w, h, _ = frame.shape
        ee = np.zeros(frame.shape)

        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(imgray, 200, 255, 0)
        edges = cv2.Canny(imgray, 150, 210, L2gradient=True)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        closed_contours = []
        for n, i in enumerate(contours):
            if cv2.contourArea(i) > cv2.arcLength(i, True) and cv2.contourArea(i) > (
                    w / 1080) ** 2 * 15000 and n % 2 == 0 and cv2.contourArea(i) < 10000:
                closed_contours.append(i)

        # Filter other bbox.
        width_list = np.asarray([np.max(i[:, :, 0]) - np.min(i[:, :, 0]) for i in closed_contours]).reshape(-1,
                                                                                                            1)
        final_predict = ""
        finger_predict = ""

        if len(width_list) != 0:
            if len(width_list) == 1:
                kmeans = KMeans(n_clusters=1, random_state=0).fit(width_list)
            else:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(width_list)

            kmeans_labels = kmeans.labels_.tolist()
            flag = None

            if kmeans_labels.count(0) > 1: flag = 0
            if kmeans_labels.count(1) > 1: flag = 1

            if kmeans_labels.count(0) > kmeans_labels.count(1):
                max_count = kmeans_labels.count(0)
            else:
                max_count = kmeans_labels.count(1)

            idx = np.where(kmeans.labels_ == flag)[0].tolist()
            filtered_contours = [con for i, con in enumerate(closed_contours) if i in idx]
            # print('**NEW FRAME********************************************************')

            finger_pos = -1

            # for con_id, con in enumerate(filtered_contours):
            for con_id, con in reversed(list(enumerate(reversed(filtered_contours)))):

                button_num = con_id + 1

                # Every screen
                xmin = np.min(con[:, :, 0])
                xmax = np.max(con[:, :, 0])
                ymin = np.min(con[:, :, 1])
                ymax = np.max(con[:, :, 1])

                button_xmin = int(xmin - 0.3 * (xmax - xmin))
                button_xmax = int(xmax - 1.16 * (xmax - xmin))
                button_ymin = int(ymin + 0.15 * (ymax - ymin))
                button_ymax = int(ymax - 0.11 * (ymax - ymin))

                # The center pixel of the button box
                center_x = button_xmin + int(0.5 * (button_xmax - button_xmin))
                center_y = button_ymin + int(0.5 * (button_ymax - button_ymin))

                # The center pixel of the button box (upper to this button box)
                upper_center_x = center_x
                upper_center_y = center_y - int(2.1 * (button_ymax - button_ymin))

                button = False
                # print("ID is:", con_id)
                if button_num != 0:

                    # print("Analying button No.", button_num)

                    my_center_R = frame[center_y, center_x, 2]
                    up_center_R = frame[upper_center_y, upper_center_x, 2]

                    if my_center_R > up_center_R:
                        diff = my_center_R - up_center_R
                    else:
                        diff = up_center_R - my_center_R

                    if diff > 40:
                        button = True
                        finger_pos = button_num  # Finger is on this button
                    else:
                        button = False

                cv2.rectangle(frame, (button_xmin, button_ymin), (button_xmax, button_ymax), (0, 255, 0), 2)

                region = np.array(frame[ymin + 5:ymax - 5, xmin + 5:xmax - 5])
                bboxes_text, polys_text, score_text = test_net(net, region,
                                                               args.text_threshold, args.link_threshold,
                                                               args.low_text, args.cuda, args.poly, refine_net)

                if len(bboxes_text) != 0:
                    image_tensors = []
                    transform = ResizeNormalize((args.imgW, args.imgH))
                    for bbox in bboxes_text:
                        xxmin = int(np.min(bbox[:, 0]))
                        xxmax = int(np.max(bbox[:, 0]))
                        yymin = int(np.min(bbox[:, 1]))
                        yymax = int(np.max(bbox[:, 1]))

                        if xxmin < 0:
                            xxmin = 0
                        if xxmax < 0:
                            xxmax = 0
                        if yymin < 0:
                            yymin = 0
                        if yymax < 0:
                            yymax = 0

                        roi = np.array(region[yymin:yymax, xxmin:xxmax])
                        roi_pil = Image.fromarray(roi).convert('L')
                        image_tensors.append(transform(roi_pil))

                    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
                    predict_list = demo(opt=args, roi=image_tensors, button=button)

                    # Special Cases

                    predict_string = ' '.join(predict_list)

                    if predict_string == "model load":
                        predict_string = "load model"

                    # Find the corresponding string item based on the predicted string

                    if button_num == 1:
                        scores = []
                        for item in words1:
                            scores.append(similar(item, predict_string))

                        predict_string = words1[scores.index(max(scores))]

                    if button_num == 2:
                        scores = []
                        for item in words2:
                            scores.append(similar(item, predict_string))

                        predict_string = words2[scores.index(max(scores))]

                    if button_num == 3:
                        scores = []
                        for item in words3:
                            scores.append(similar(item, predict_string))

                        predict_string = words3[scores.index(max(scores))]

                    if button_num == 4:
                        scores = []
                        for item in words4:
                            scores.append(similar(item, predict_string))

                        predict_string = words4[scores.index(max(scores))]

                    final_predict = final_predict + ";" + predict_string

                    for bbox in bboxes_text:
                        bbox[:, 0] = bbox[:, 0] + xmin
                        bbox[:, 1] = bbox[:, 1] + ymin

                        poly = np.array(bbox).astype(np.int32).reshape((-1))
                        poly = poly.reshape(-1, 2)
                        cv2.polylines(frame, [poly.reshape((-1, 1, 2))], True, (0, 0, 255), 2)

                else:

                    final_predict = final_predict + ";" + "EMPTY STRING"

        duration = time.time() - start_time

        grid = cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 3)



        print("Duration: ", duration)
        print(f"[Image No.{counter}]: ", final_predict)

        cv2.imshow('image', grid)
        cv2.waitKey(0)
        cv2.imwrite("test.jpg", frame)
        counter += 1


if __name__ == "__main__":
    print("Finger-text detection testing...")
    worker_predict_image()
