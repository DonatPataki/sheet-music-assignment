import cv2 as cv
import numpy as np

def main():
    padding_x = 7
    padding_y = 7
    line_padding = 40

    img = cv.imread('img/1.jpg')

    original = img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(img, (5, 5), 0)
    (thresh, bw_image) = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)

    height = img.shape[0]
    width = img.shape[1]

    corners = [(0, 0)] * 4

    def getCornerPoint(y_range, x_range):
        index = 0
        y = y_range[0]
        for x in x_range:
            if bw_image[y, x] == 255:
                index = x
                break
        
        prev_pos = (0, 0)
        for y in y_range:
            for x in x_range:
                if bw_image[y, x] == 255:
                    prev_index = index
                    index = x
                    if abs(prev_index - index) > 1:
                        return prev_pos
                    else:
                        prev_pos = (x, y)
                        break

    corners[0] = getCornerPoint(range(height // 3, 0, -1), range(0, width // 3))
    corners[1] = getCornerPoint(range(height // 3, 0, -1), range(width - 1, width - width // 3, -1))
    corners[2] = getCornerPoint(range(height - height // 3, height), range(0, width // 3))
    corners[3] = getCornerPoint(range(height - height // 3, height), range(width - 1, width - width // 3, -1))

    original_img_copy = original.copy()

    cv.line(original_img_copy, (corners[0][0] + padding_x, corners[0][1] + padding_y), (corners[1][0] - padding_x, corners[1][1] + padding_y), (0, 0, 255), 1)
    cv.line(original_img_copy, (corners[2][0] + padding_x, corners[2][1] - padding_y), (corners[3][0] - padding_x, corners[3][1] - padding_y), (0, 0, 255), 1)
    cv.line(original_img_copy, (corners[0][0] + padding_x, corners[0][1] + padding_y), (corners[2][0] + padding_x, corners[2][1] - padding_y), (0, 0, 255), 1)
    cv.line(original_img_copy, (corners[1][0] - padding_x, corners[1][1] + padding_y), (corners[3][0] - padding_x, corners[3][1] - padding_y), (0, 0, 255), 1)

    cv.imwrite("1.jpg", original_img_copy)

    crop_points = np.array([[corners[0][0] + padding_x, corners[0][1] + padding_y], [corners[1][0] - padding_x, corners[1][1] + padding_y], [corners[3][0] - padding_x, corners[3][1] - padding_y], [corners[2][0] + padding_x, corners[2][1] - padding_y]])

    mask = np.zeros(original.shape[:2], np.uint8)
    cv.drawContours(mask, [crop_points], -1, (255, 255, 255), -1, cv.LINE_AA)
    dest = cv.bitwise_and(original, original, mask=mask)
    bg = np.ones_like(original, np.uint8) * 255
    cv.bitwise_not(bg, bg, mask=mask)
    crop = bg + dest

    cv.imwrite("2.jpg", crop)

    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    (thresh, gray) = cv.threshold(blur, 155, 255, cv.THRESH_BINARY)
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cv.imwrite("3.jpg", bw)

    cols = horizontal.shape[1]
    horizontal_size = cols // 35
    kernel = np.ones((1, horizontal_size), np.uint8)
    horizontal = cv.erode(horizontal, kernel)
    horizontal = cv.dilate(horizontal, kernel)

    kernel = np.ones((1, 250), np.uint8)
    horizontal = cv.dilate(horizontal, kernel, iterations=1)
    horizontal = cv.erode(horizontal, kernel, iterations=1)

    rows = vertical.shape[0]
    vertical_size = rows // 30
    kernel = np.ones((vertical_size, 1), np.uint8)
    vertical = cv.erode(vertical, kernel)
    vertical = cv.dilate(vertical, kernel)

    zeros = np.zeros(height)
    col_sums = np.sum(vertical, axis=0)

    temp = []
    result = []
    for x in range(width):
        if col_sums[x] > 0:
            if len(temp) > 0 and x - temp[-1] != 1:
                result.append(temp)
                temp = []
            temp.append(x)
    result.append(temp)

    temp1 = result.pop(1)
    temp2 = result.pop(-2)

    r1 = range(temp1[1], temp1[1] + 50)
    r2 = range(temp2[-1] - 50, temp2[-1])

    result = []
    for x in range(width):
        if x not in r1 and x not in r2:
            result.append(x)

    for x in range(width):
        if x in result:
            vertical[:, x] = zeros


    zeros = np.zeros(width)
    row_sums = np.sum(horizontal, axis=1)

    temp = []
    result = []
    for y in range(height):
        if row_sums[y] > 0:
            if len(temp) > 0 and y - temp[-1] != 1:
                result.append(temp)
                temp = []
            temp.append(y)
    result.append(temp)

    temp1 = result.pop()
    temp2 = result.pop(0)

    r1 = range(temp1[0], temp1[-1] + 1)
    r2 = range(temp2[0], temp2[-1] + 1)

    result = []
    for y in range(height):
        if y not in r1 and y not in r2:
            result.append(y)

    for y in range(height):
        if y not in result:
            horizontal[y, :] = zeros


    lines_mask = vertical + horizontal

    kernel = np.ones((1, 200), np.uint8)
    lines_mask = cv.dilate(lines_mask, kernel, iterations=1)
    lines_mask = cv.erode(lines_mask, kernel, iterations=1)

    kernel = np.ones((1, 20), np.uint8)
    lines_mask = cv.erode(lines_mask, kernel, iterations=1)

    kernel = np.ones((1, 10), np.uint8)
    lines_mask = cv.dilate(lines_mask, kernel, iterations=1)

    kernel = np.ones((3, 1), np.uint8)
    lines_mask = cv.dilate(lines_mask, kernel, iterations=1)

    row_sums = np.sum(horizontal, axis=1)
    zeros = np.zeros(width)

    cv.imwrite("4.jpg", lines_mask)

    col_sums = np.sum(horizontal, axis=0)

    begin = np.argmax(col_sums > 255 * 15)

    end = len(col_sums) - np.argmax(col_sums[::-1] > 255 * 15) - 1

    temp = []

    def check_line():
        nonlocal temp
        nonlocal lines_mask
        
        reached_bottom = False
        points = []
        center = temp[len(temp) // 2]
        checking = True
        offset = [center, width // 2]
        skip_left = False
        while checking:
            if lines_mask[offset[0]][offset[1] + 1] == 255:
                offset[1] += 1
            elif lines_mask[offset[0] + 1][offset[1]] == 255 and not reached_bottom:
                offset[0] += 1
            elif lines_mask[offset[0] - 1][offset[1]] == 255:
                reached_bottom = True
                offset[0] -= 1
            else:
                checking = False
            points.append(offset.copy())
        if points[-1][1] < end - line_padding:
            skip_left = True
        reached_bottom = False
        checking = True
        center = temp[len(temp) // 2]
        offset = [center, width // 2]
        if not skip_left:
            while checking:
                if lines_mask[offset[0]][offset[1] - 1] == 255:
                    offset[1] -= 1
                elif lines_mask[offset[0] + 1][offset[1]] == 255 and not reached_bottom:
                    offset[0] += 1
                elif lines_mask[offset[0] - 1][offset[1]] == 255:
                    reached_bottom = True
                    offset[0] -= 1
                else:
                    checking = False
                points.append(offset.copy())
            temp = []
            if points[-1][1] > begin + line_padding:
                points = []
            for point in points:
                original[point[0]][point[1]] = [0, 0, 255]
            points = []
            temp.append(y)

    for y in range(height):
        if horizontal[y][width // 2] == 255:
            if len(temp) == 0:
                temp.append(y)
            elif y - temp[-1] == 1:
                temp.append(y)
            else:
                check_line()
                temp = []
        elif y == height -1:
            check_line()
            temp = []

    cv.imwrite("5.jpg", original)

    k = cv.waitKey(0)

if __name__ == "__main__":
    main()
